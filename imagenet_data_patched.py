# imagenet_data_patched.py
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Reuse dataset utilities from imagenet_data.py
from imagenet_data import load_wnid_to_idx, ImageNetTrainFolder

# -------------------------------------------------------------------------
# Patch color targets (3-bit mean color per patch -> 512-way classification)
# -------------------------------------------------------------------------

def compute_patch_color_labels(
    img_tensor: torch.Tensor,
    patch_size: int = 16,
    bits: int = 3,
) -> torch.Tensor:
    """
    Compute a 3-bit-per-channel mean color label for each patch.

    Args:
        img_tensor: Tensor of shape (3, H, W) with values in [0, 1].
        patch_size: Patch size (16 for ViT-B/16).
        bits: Number of bits per channel (3 bits -> 8 bins per channel).

    Returns:
        labels: LongTensor of shape (num_patches,) with values in [0, 512).
                Patch order matches ViT patch embedding order (row-major).
    """
    assert img_tensor.dim() == 3, "Expected (C, H, W) tensor."
    C, H, W = img_tensor.shape
    assert C == 3, "Expected 3 channels (RGB)."
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size."

    Hp = H // patch_size
    Wp = W // patch_size

    # Reshape into (C, Hp, patch_size, Wp, patch_size)
    x = img_tensor.reshape(C, Hp, patch_size, Wp, patch_size)
    # Mean over patch spatial dimensions -> (C, Hp, Wp)
    patch_means = x.mean(dim=(2, 4))

    # (Hp, Wp, C) then flatten to (num_patches, 3)
    patch_means = patch_means.permute(1, 2, 0).reshape(-1, 3)

    # Quantize each channel to 2**bits bins in [0, 2**bits - 1]
    bins = 2 ** bits
    # Clamp just in case of numerical ~1.0
    patch_means = patch_means.clamp(0.0, 1.0 - 1e-8)
    q = (patch_means * bins).long().clamp(0, bins - 1)  # (N, 3)

    r, g, b = q[:, 0], q[:, 1], q[:, 2]
    # Pack into a single 3*bits-bit integer: R * 2^(2*bits) + G * 2^bits + B
    labels = (r << (2 * bits)) + (g << bits) + b  # in [0, 512)
    return labels  # (num_patches,)


# -------------------------------------------------------------------------
# Dataset wrapper for masked patch prediction pre-training
# -------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetTrainPatched(ImageNetTrainFolder):
    """
    ImageNet training set that, in addition to the image and class label,
    returns patch-wise color targets for masked patch prediction.

    __getitem__ returns:
        img_norm: normalized tensor (3, 224, 224)
        patch_labels: LongTensor (num_patches,) in [0, 512)
        cls_target: int in [0, 999] (supervised label; can be ignored in self-supervision)
    """

    def __init__(
        self,
        root,
        wnid_to_idx,
        image_transform=None,
        normalize=True,
        patch_size: int = 16,
        color_bits: int = 3,
    ):
        super().__init__(root, wnid_to_idx, transform=None)

        # Spatial / geometric + ToTensor transforms (no normalization)
        if image_transform is None:
            image_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # [0,1]
                ]
            )

        self.image_transform = image_transform
        self.patch_size = patch_size
        self.color_bits = color_bits

        self.normalize = normalize
        self._norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalize else None

    def __getitem__(self, idx):
        path, cls_target = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # 1) Apply spatial + ToTensor transforms -> (3, 224, 224) in [0,1]
        img_tensor = self.image_transform(img)

        # 2) Compute patch color labels BEFORE normalization
        patch_labels = compute_patch_color_labels(
            img_tensor,
            patch_size=self.patch_size,
            bits=self.color_bits,
        )

        # 3) Optionally normalize for ViT
        if self._norm is not None:
            img_tensor = self._norm(img_tensor)

        return img_tensor, patch_labels, cls_target


def build_imagenet_mpp_train_loader(
    train_root="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    devkit_root=os.path.expanduser("~/connected-pixels/ILSVRC2012_devkit_t12"),
    train_batch_size=256,
    num_workers=8,
    patch_size: int = 16,
    color_bits: int = 3,
):
    """
    Build a DataLoader for masked patch prediction pre-training.

    Returns:
        train_loader, wnid_to_idx

    Each batch from train_loader is:
        images:       (B, 3, 224, 224)
        patch_labels: (B, num_patches)  # num_patches = (224/patch_size)^2
        cls_targets:  (B,)              # standard ImageNet label (optional)
    """
    meta_path = os.path.join(devkit_root, "data", "meta.mat")
    wnid_to_idx = load_wnid_to_idx(meta_path)

    dataset = ImageNetTrainPatched(
        train_root,
        wnid_to_idx,
        image_transform=None,  # use default defined in __init__
        normalize=True,
        patch_size=patch_size,
        color_bits=color_bits,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, wnid_to_idx


# -------------------------------------------------------------------------
# Masking procedure for patch embeddings (B.1.2 in ViT paper)
# -------------------------------------------------------------------------

class PatchMasker:
    """
    Implements the masked patch prediction corruption for ViT patch embeddings.

    Given patch embeddings (e.g. output of vit.encoder after patch embedding + pos. emb.,
    or directly after the patch embedding if you prefer), it:

      - Selects 50% of patch positions (mask_ratio) per sample.
      - For those positions:
          80% -> replace embedding with a learnable [MASK] embedding.
          10% -> replace embedding with a random other patch embedding.
          10% -> keep embedding as is (but still predict its color).

    This mimics BERT-style masking and the setup described in Appendix B.1.2. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        embed_dim: int,
        mask_ratio: float = 0.5,
        p_mask: float = 0.8,
        p_random: float = 0.1,
        p_keep: float = 0.1,
        device=None,
    ):
        assert abs(p_mask + p_random + p_keep - 1.0) < 1e-6, "Probabilities must sum to 1."

        self.mask_ratio = mask_ratio
        self.p_mask = p_mask
        self.p_random = p_random
        self.p_keep = p_keep

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if device is not None:
            self.mask_token.data = self.mask_token.data.to(device)

    def to(self, device):
        self.mask_token.data = self.mask_token.data.to(device)
        return self

    @torch.no_grad()
    def _sample_corrupt_positions(self, num_patches: int, device: torch.device):
        """
        Sample indices of patches to corrupt for a single sample.

        Returns:
            idx: LongTensor (M,) with M ≈ mask_ratio * num_patches
        """
        num_to_corrupt = int(round(self.mask_ratio * num_patches))
        if num_to_corrupt == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        perm = torch.randperm(num_patches, device=device)
        return perm[:num_to_corrupt]

    def apply(self, patch_embeddings: torch.Tensor):
        """
        Apply corruption to patch embeddings.

        Args:
            patch_embeddings: Tensor of shape (B, L, D)
                L = 1 + N for ViT (CLS token + N patches).
                CLS token at index 0 is never masked.

        Returns:
            corrupted:       (B, L, D) corrupted embeddings
            is_corrupted:    (B, N) bool mask, True for patches we predict
            corruption_type: (B, N) long, 0=not selected, 1=mask, 2=random, 3=keep
        """
        B, L, D = patch_embeddings.shape
        assert L >= 2, "Expect CLS + at least one patch."
        N = L - 1  # number of patch tokens
        device = patch_embeddings.device

        corrupted = patch_embeddings.clone()
        is_corrupted = torch.zeros(B, N, dtype=torch.bool, device=device)
        corruption_type = torch.zeros(B, N, dtype=torch.long, device=device)

        probs = torch.tensor([self.p_mask, self.p_random, self.p_keep], device=device)

        for b in range(B):
            idx = self._sample_corrupt_positions(N, device)
            if idx.numel() == 0:
                continue

            # Mark these as corrupted
            is_corrupted[b, idx] = True

            # Sample type for each selected patch: 0=mask,1=random,2=keep
            g = torch.multinomial(probs, num_samples=idx.numel(), replacement=True)
            # Store type+1 so that 0 can mean "not selected"
            corruption_type[b, idx] = g + 1

            original_patches = patch_embeddings[b, 1:, :]  # (N, D)
            patch_slice = corrupted[b, 1:, :]               # (N, D)

            # -- MASK --
            mask_positions = idx[g == 0]
            if mask_positions.numel() > 0:
                mask_token_2d = self.mask_token.view(1, D).expand(mask_positions.numel(), D)
                patch_slice[mask_positions] = mask_token_2d

            # -- RANDOM OTHER PATCH --
            random_positions = idx[g == 1]
            if random_positions.numel() > 0:
                # sample random source indices in [0, N)
                src = torch.randint(
                    low=0,
                    high=N,
                    size=(random_positions.numel(),),
                    device=device,
                )
                # Optionally ensure "other patch" by avoiding same index
                same = src == random_positions
                if same.any():
                    src[same] = (src[same] + 1) % N
                patch_slice[random_positions] = original_patches[src]

            # -- KEEP (embeddings unchanged, but still to be predicted) --
            keep_positions = idx[g == 2]
            if keep_positions.numel() > 0:
                patch_slice[keep_positions] = original_patches[keep_positions]

            corrupted[b, 1:, :] = patch_slice

        return corrupted, is_corrupted, corruption_type


def build_mpp_targets(patch_color_labels: torch.Tensor,
                      is_corrupted: torch.Tensor,
                      ignore_index: int = -100) -> torch.Tensor:
    """
    Given per-patch color labels and a corruption mask, build targets for
    cross-entropy loss where only corrupted patches contribute.

    Args:
        patch_color_labels: (B, N) LongTensor in [0, 512)
        is_corrupted:       (B, N) BoolTensor
        ignore_index:       label to ignore in loss

    Returns:
        targets: (B, N) LongTensor, with non-corrupted positions set to ignore_index.
    """
    assert patch_color_labels.shape == is_corrupted.shape
    targets = torch.full_like(patch_color_labels, ignore_index)
    targets[is_corrupted] = patch_color_labels[is_corrupted]
    return targets


# -------------------------------------------------------------------------
# Tiny usage sketch (not executed if imported)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing ImageNet MPP train loader and PatchMasker...")

    train_loader, wnid_to_idx = build_imagenet_mpp_train_loader(
        train_batch_size=8,
        num_workers=2,
    )

    images, patch_labels, cls_targets = next(iter(train_loader))
    print("Images:", images.shape)
    print("Patch labels:", patch_labels.shape)
    print("Class targets:", cls_targets.shape)

    # Example: apply masking to dummy ViT patch embeddings
    B, C, H, W = images.shape
    embed_dim = 768  # ViT-B/16
    num_patches = (H // 16) * (W // 16)
    dummy_embeddings = torch.randn(B, 1 + num_patches, embed_dim)

    masker = PatchMasker(embed_dim=embed_dim)
    corrupted, is_corr, corr_type = masker.apply(dummy_embeddings)

    print("Corrupted embeddings:", corrupted.shape)
    print("is_corrupted:", is_corr.shape, "num corrupted:", is_corr.sum().item())

    patch_labels = patch_labels.view(B, num_patches)
    mpp_targets = build_mpp_targets(patch_labels, is_corr, ignore_index=-100)
    print("MPP targets shape:", mpp_targets.shape)
