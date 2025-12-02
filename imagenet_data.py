import os
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
from scipy.io import loadmat

def load_wnid_to_idx(meta_path):
    """
    Returns a dict: wnid (e.g. 'n01440764') -> class index in [0, 999],
    consistent with ILSVRC2012 IDs and the devkit.
    """
    meta = loadmat(meta_path, squeeze_me=True)
    synsets = meta["synsets"]  # array of entries; each entry m: [ILSVRC2012_ID, WNID, ...]
    ilsvrc2012_id_to_wnid = {int(m[0]): str(m[1]) for m in synsets}

    # Keep only the 1000 classification classes
    wnid_to_idx = {
        wnid: ilsvrc_id - 1  # make it 0-based
        for ilsvrc_id, wnid in ilsvrc2012_id_to_wnid.items()
        if 1 <= ilsvrc_id <= 1000
    }
    return wnid_to_idx

class ImageNetTrainFolder(Dataset):
    def __init__(self, root, wnid_to_idx, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        # Each subfolder is a wnid like n01440764
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            wnid = class_dir.name
            if wnid not in wnid_to_idx:
                # Some entries in meta.mat are unused; skip them
                continue
            target = wnid_to_idx[wnid]
            # Official files are *.JPEG; you can be more permissive if needed
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

class ImageNetValFromDevkit(Dataset):
    """
    Uses the original 50k validation images as a 'test' set.

    Labels come from ILSVRC2012_validation_ground_truth.txt
    and are mapped to [0, 999] via (id - 1), consistent with wnid_to_idx.
    """
    def __init__(self, val_root, devkit_root, transform=None):
        self.val_root = Path(val_root)
        self.transform = transform

        gt_path = Path(devkit_root) / "data" / "ILSVRC2012_validation_ground_truth.txt"
        with open(gt_path, "r") as f:
            # Each line is a 1-based ILSVRC2012_ID
            self.ilsvrc_ids = [int(line.strip()) for line in f if line.strip()]

        self.samples = []
        for i, ilsvrc_id in enumerate(self.ilsvrc_ids, start=1):
            fname = f"ILSVRC2012_val_{i:08d}.JPEG"
            path = self.val_root / fname
            target = ilsvrc_id - 1  # 0-based index
            self.samples.append((str(path), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def build_imagenet_loaders(
    train_root="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    val_root="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
    devkit_root=os.path.expanduser("~/connected-pixels/ILSVRC2012_devkit_t12"),
    train_batch_size=256,
    test_batch_size=256,
    num_workers=8,
    val_fraction=0.1,
):
    meta_path = os.path.join(devkit_root, "data", "meta.mat")
    wnid_to_idx = load_wnid_to_idx(meta_path)

    # Transforms (you can tweak to match standard ImageNet recipes)
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

    # Full train dataset
    full_train = ImageNetTrainFolder(train_root, wnid_to_idx, transform=train_transform)

    # Split into train / val (both supervised)
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    # For validation we want eval transforms, not train transforms:
    # random_split keeps a reference to full_train, so we override transforms:
    full_train.transform = train_transform
    val_dataset.dataset.transform = eval_transform

    # Test set = original ImageNet val, labeled via devkit
    test_dataset = ImageNetValFromDevkit(val_root, devkit_root, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, wnid_to_idx



if __name__ == "__main__":
    print("Testing the data loaders.")

    train_loader, val_loader, test_loader, wnid_to_idx = build_imagenet_loaders()

    print("Loaded training images.")
    num_batches = len(train_loader)
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        print(f"Batch {batch_idx}/{num_batches}")
        break

    print("Loaded validation images.")
    num_batches = len(val_loader)
    for batch_idx, (images, labels) in enumerate(val_loader, start=1):
        print(f"Batch {batch_idx}/{num_batches}")
        break

    print("Loaded test images.")
    num_batches = len(test_loader)
    for batch_idx, (images, labels) in enumerate(test_loader, start=1):
        print(f"Batch {batch_idx}/{num_batches}")
        break
