# all_data_masked.py
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from imagenet_data_masked import (
    load_wnid_to_idx,
    ImageNetTrainFolder,
    ImageNetMaskedPatchDataset,
)


class Coco2017ImageFolder(Dataset):
    """
    Minimal COCO image dataset for masked patch prediction.
    Ignores annotations; returns (image, dummy_label) to match ImageNet wrapper API.
    """
    def __init__(self, images_root, transform=None, dummy_label=0):
        self.root = Path(images_root)
        self.transform = transform
        self.dummy_label = int(dummy_label)

        if not self.root.exists():
            raise FileNotFoundError(f"COCO images_root not found: {self.root}")

        self.samples = sorted(self.root.glob("*.jpg"))
        if len(self.samples) == 0:
            raise RuntimeError(f"No .jpg files found under: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.dummy_label


class EvenInterleavedDataset(Dataset):
    """
    Acts like a single dataset of length len(A)+len(B), with items from B
    spread out as evenly as possible through A (proportional interleaving).

    It also supports per-epoch reshuffling *within* A and within B while
    preserving the global even spread.

    NOTE: If you use DataLoader(num_workers>0), call set_epoch(epoch)
    BEFORE you create the iterator (i.e., before for-batch loop each epoch).
    Keep persistent_workers=False (default) unless you re-create the loader.
    """
    def __init__(self, dataset_a, dataset_b, base_seed=42):
        self.a = dataset_a
        self.b = dataset_b
        self.la = len(self.a)
        self.lb = len(self.b)
        self.lt = self.la + self.lb
        self.base_seed = int(base_seed)

        if self.la <= 0 or self.lb <= 0:
            raise ValueError(f"Need non-empty datasets. Got len(A)={self.la}, len(B)={self.lb}")

        self._epoch = 0
        self._make_perms()

    def _make_perms(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self._epoch)
        self.perm_a = torch.randperm(self.la, generator=g)
        self.perm_b = torch.randperm(self.lb, generator=g)

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch != self._epoch:
            self._epoch = epoch
            self._make_perms()

    def __len__(self):
        return self.lt

    @staticmethod
    def _count_b_up_to(i: int, lb: int, lt: int) -> int:
        # number of B items in positions [0..i] (inclusive), using floor
        # ensures total at i=lt-1 equals lb
        return int(((i + 1) * lb) // lt)

    def __getitem__(self, i):
        if i < 0 or i >= self.lt:
            raise IndexError(i)

        # Decide if position i is from B by checking whether count_b increased at i
        cb_i = self._count_b_up_to(i, self.lb, self.lt)
        cb_prev = 0 if i == 0 else self._count_b_up_to(i - 1, self.lb, self.lt)

        if cb_i != cb_prev:
            # position i is from B
            rank_b = cb_i - 1
            j = int(self.perm_b[rank_b].item())
            return self.b[j]
        else:
            # position i is from A
            # number of B items up to i is cb_i, so A rank is i - cb_i
            rank_a = i - cb_i
            j = int(self.perm_a[rank_a].item())
            return self.a[j]


def build_all_masked_loaders(
    # ImageNet
    imagenet_train_root="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    imagenet_devkit_root=os.path.expanduser("~/connected-pixels/ILSVRC2012_devkit_t12"),
    imagenet_val_fraction=0.1,
    # COCO
    coco_root="data/coco2017",
    # Loader
    train_batch_size=256,
    val_batch_size=256,
    num_workers=8,
    # MPP
    patch_size=16,
    mask_ratio=0.5,
    return_label=False,
    # Combined
    base_seed=42,
):
    """
    Returns:
      combined_train_loader, combined_val_loader, wnid_to_idx, combined_train_dataset

    combined_train_dataset has .set_epoch(epoch) if you want reshuffling each epoch.
    """
    meta_path = os.path.join(imagenet_devkit_root, "data", "meta.mat")
    wnid_to_idx = load_wnid_to_idx(meta_path)

    # MUST match imagenet_data_masked.py augmentations
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

    # ---- ImageNet base + split ----
    imagenet_full = ImageNetTrainFolder(imagenet_train_root, wnid_to_idx, transform=train_transform)

    val_size = int(len(imagenet_full) * imagenet_val_fraction)
    train_size = len(imagenet_full) - val_size
    imagenet_train_subset, imagenet_val_subset = random_split(
        imagenet_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # set val transform
    imagenet_val_subset.dataset.transform = eval_transform

    imagenet_train_mpp = ImageNetMaskedPatchDataset(
        imagenet_train_subset, patch_size=patch_size, mask_ratio=mask_ratio, return_label=return_label
    )
    imagenet_val_mpp = ImageNetMaskedPatchDataset(
        imagenet_val_subset, patch_size=patch_size, mask_ratio=mask_ratio, return_label=return_label
    )

    # ---- COCO base ----
    coco_root = Path(coco_root)
    coco_train_base = Coco2017ImageFolder(coco_root / "train2017", transform=train_transform)
    coco_val_base = Coco2017ImageFolder(coco_root / "val2017", transform=eval_transform)

    coco_train_mpp = ImageNetMaskedPatchDataset(
        coco_train_base, patch_size=patch_size, mask_ratio=mask_ratio, return_label=return_label
    )
    coco_val_mpp = ImageNetMaskedPatchDataset(
        coco_val_base, patch_size=patch_size, mask_ratio=mask_ratio, return_label=return_label
    )

    # ---- Combined datasets (even spread) ----
    combined_train_dataset = EvenInterleavedDataset(imagenet_train_mpp, coco_train_mpp, base_seed=base_seed)
    combined_val_dataset = EvenInterleavedDataset(imagenet_val_mpp, coco_val_mpp, base_seed=base_seed + 999)

    # Keep shuffle=False so our even spread is preserved.
    # If you want new ordering each epoch, call combined_train_dataset.set_epoch(epoch)
    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    combined_val_loader = DataLoader(
        combined_val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return combined_train_loader, combined_val_loader, wnid_to_idx, combined_train_dataset


if __name__ == "__main__":
    train_loader, val_loader, _, train_ds = build_all_masked_loaders(
        coco_root="data/coco2017",
        train_batch_size=512,
        val_batch_size=512,
        num_workers=8,
        imagenet_val_fraction=0.1,
    )

    # example: reshuffle each "epoch"
    for epoch in range(2):
        train_ds.set_epoch(epoch)
        batch = next(iter(train_loader))
        print("epoch", epoch, [x.shape for x in batch])
