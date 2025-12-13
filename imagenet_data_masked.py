# imagenet_data_masked.py
import os
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from scipy.io import loadmat


def load_wnid_to_idx(meta_path):
    """
    Returns a dict: wnid (e.g. 'n01440764') -> class index in [0, 999],
    consistent with ILSVRC2012 IDs and the devkit.
    """
    meta = loadmat(meta_path, squeeze_me=True)
    synsets = meta["synsets"]  # each entry m: [ILSVRC2012_ID, WNID, ...]
    ilsvrc2012_id_to_wnid = {int(m[0]): str(m[1]) for m in synsets}

    # Keep only the 1000 classification classes
    wnid_to_idx = {
        wnid: ilsvrc_id - 1  # make it 0-based
        for ilsvrc_id, wnid in ilsvrc2012_id_to_wnid.items()
        if 1 <= ilsvrc_id <= 1000
    }
    return wnid_to_idx


class ImageNetTrainFolder(Dataset):
    """
    Train images live in:
      train_root/<wnid>/*.JPEG
    where <wnid> is something like 'n01440764'.

    Even for masked-patch pretraining we keep the class index around
    for potential debugging or later reuse, but the wrapper dataset
    can choose to drop it.
    """
    def __init__(self, root, wnid_to_idx, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            wnid = class_dir.name
            if wnid not in wnid_to_idx:
                continue
            target = wnid_to_idx[wnid]
            for img_path in class_dir.glob("*.JPEG"):
                self.samples.append((str(img_path), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _patchify(img_tensor, patch_size):
    """
    img_tensor: [3, H, W] (float in [0,1] after ToTensor)
    returns patches: [N, 3, P, P]
    """
    if img_tensor.dim() != 3 or img_tensor.size(0) != 3:
        raise ValueError(f"Expected img_tensor [3,H,W], got {tuple(img_tensor.shape)}")

    c, h, w = img_tensor.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size {(h,w)} not divisible by patch_size={patch_size}")

    gh = h // patch_size
    gw = w // patch_size

    # [3, gh, P, gw, P] -> [gh, gw, 3, P, P] -> [N, 3, P, P]
    patches = (
        img_tensor.view(c, gh, patch_size, gw, patch_size)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
        .view(gh * gw, c, patch_size, patch_size)
    )
    return patches


def mean_3bit_color_targets(img_tensor, patch_size):
    """
    Compute per-patch mean RGB and quantize each channel to 3-bit (0..7).

    Returns:
      targets: LongTensor [N, 3] with values in [0..7]
    """
    patches = _patchify(img_tensor, patch_size)  # [N, 3, P, P]
    mean_rgb = patches.mean(dim=(2, 3))          # [N, 3] in [0,1]
    q = torch.clamp((mean_rgb * 7.0).round(), 0.0, 7.0).to(torch.long)
    return q


def sample_patch_mask(num_patches, mask_ratio=0.5, device=None):
    """
    Returns a boolean mask of shape [num_patches] with approximately
    mask_ratio fraction True. (We use round() for a stable count.)
    """
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError("mask_ratio must be in [0,1]")

    n_mask = int(round(num_patches * mask_ratio))
    perm = torch.randperm(num_patches, device=device)
    mask_idx = perm[:n_mask]
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[mask_idx] = True
    return mask


class ImageNetMaskedPatchDataset(Dataset):
    """
    Wraps a base dataset that returns (img, class_idx) and outputs:
      img: Tensor [3,224,224]
      mask: BoolTensor [N] indicating which patches should be corrupted
      targets: LongTensor [N,3] mean 3-bit color targets
      class_idx: int (optional; kept for debugging/reuse)
    """
    def __init__(self, base_dataset, patch_size=16, mask_ratio=0.5, return_label=False):
        self.base = base_dataset
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.return_label = return_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, cls = self.base[idx]
        targets = mean_3bit_color_targets(img, self.patch_size)
        mask = sample_patch_mask(targets.size(0), self.mask_ratio, device=None)
        if self.return_label:
            return img, mask, targets, cls
        return img, mask, targets


def build_imagenet_masked_loaders(
    train_root="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    devkit_root=os.path.expanduser("~/connected-pixels/ILSVRC2012_devkit_t12"),
    train_batch_size=256,
    val_batch_size=256,
    num_workers=8,
    val_fraction=0.1,
    patch_size=16,
    mask_ratio=0.5,
    return_label=False,
):
    """
    Builds loaders for masked patch prediction pretraining using the same
    ImageNet train folder and the same transforms as imagenet_data.py.

    Returns:
      pretrain_train_loader, pretrain_val_loader, wnid_to_idx
    """
    meta_path = os.path.join(devkit_root, "data", "meta.mat")
    wnid_to_idx = load_wnid_to_idx(meta_path)

    # IMPORTANT: keep augmentations identical to imagenet_data.py
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    full_train = ImageNetTrainFolder(train_root, wnid_to_idx, transform=train_transform)

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    train_subset, val_subset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # random_split returns Subset objects referencing full_train
    # We'll set val to eval transforms for consistency with imagenet_data.py
    full_train.transform = train_transform
    val_subset.dataset.transform = eval_transform

    pretrain_train = ImageNetMaskedPatchDataset(
        train_subset,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        return_label=return_label,
    )
    pretrain_val = ImageNetMaskedPatchDataset(
        val_subset,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        return_label=return_label,
    )

    pretrain_train_loader = DataLoader(
        pretrain_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    pretrain_val_loader = DataLoader(
        pretrain_val,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return pretrain_train_loader, pretrain_val_loader, wnid_to_idx


if __name__ == "__main__":
    train_loader, val_loader, wnid_to_idx = build_imagenet_masked_loaders(
        train_batch_size=8,
        val_batch_size=8,
        num_workers=2,
        val_fraction=0.001,  # tiny split for a quick sanity check
        patch_size=16,
        mask_ratio=0.5,
        return_label=True,
    )

    print("Loaded masked pretraining train loader.")
    images, masks, targets, cls = next(iter(train_loader))
    print("images:", images.shape)    # [B,3,224,224]
    print("masks:", masks.shape)      # [B,N]
    print("targets:", targets.shape)  # [B,N,3]
    print("cls:", cls.shape)          # [B]
