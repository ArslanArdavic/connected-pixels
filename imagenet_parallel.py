"""
Distributed-ready ImageNet data loaders.

This mirrors the splitting logic from imagenet_data.py (90% train / 10% val
from the training root, plus the official validation images as a test set),
but wraps each split in DistributedSampler so that training can be sharded
across multiple GPUs / processes.
"""
import os

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from imagenet_data import (
    ImageNetTrainFolder,
    ImageNetValFromDevkit,
    load_wnid_to_idx,
)


def _dist_info():
    """Return (rank, world_size), defaulting to (0, 1) when not initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _maybe_print(msg, rank_only=True):
    """Print from rank 0 by default to avoid noisy logs."""
    rank, _ = _dist_info()
    if (rank_only and rank == 0) or not rank_only:
        print(msg, flush=True)


def build_imagenet_distributed_loaders(
    train_root="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    val_root="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
    devkit_root=os.path.expanduser("~/connected-pixels/ILSVRC2012_devkit_t12"),
    train_batch_size=256,  # per GPU when distributed
    test_batch_size=256,  # per GPU when distributed
    num_workers=8,
    val_fraction=0.1,
    distributed=True,
    seed=42,
    drop_last=True,
):
    """
    Build ImageNet dataloaders suitable for DistributedDataParallel.

    Returns:
        train_loader, val_loader, test_loader, (train_sampler, val_sampler, test_sampler), wnid_to_idx
    """
    rank, world_size = _dist_info()
    _maybe_print(
        f"[imagenet_parallel] Building loaders | distributed={distributed} | rank={rank} | world_size={world_size}"
    )

    wnid_to_idx = load_wnid_to_idx(os.path.join(devkit_root, "data", "meta.mat"))
    _maybe_print(f"[imagenet_parallel] Loaded wnid_to_idx with {len(wnid_to_idx)} classes")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    full_train = ImageNetTrainFolder(train_root, wnid_to_idx, transform=train_transform)
    gen = torch.Generator().manual_seed(seed)
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=gen)
    val_dataset.dataset.transform = eval_transform  # swap to eval transforms

    test_dataset = ImageNetValFromDevkit(val_root, devkit_root, transform=eval_transform)

    _maybe_print(
        "[imagenet_parallel] Dataset sizes before sharding: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if distributed and world_size > 1:
        _maybe_print(
            "[imagenet_parallel] Sharded sampler lengths (approx): "
            f"train={len(train_sampler)}, val={len(val_sampler)}, test={len(test_sampler)}"
        )
    _maybe_print(
        "[imagenet_parallel] DataLoader batch sizes (per process): "
        f"train={train_batch_size}, val/test={test_batch_size}"
    )

    return train_loader, val_loader, test_loader, (train_sampler, val_sampler, test_sampler), wnid_to_idx


def set_epoch_for_samplers(epoch, samplers):
    """Call once per epoch so DistributedSampler reshuffles consistently."""
    for sampler in samplers:
        if sampler is not None:
            sampler.set_epoch(epoch)
    _maybe_print(f"[imagenet_parallel] Set sampler epoch={epoch}")


def init_distributed(local_rank, backend="nccl"):
    """
    Helper to initialize torch.distributed for multi-GPU training.
    Assumes launch via torchrun which sets LOCAL_RANK.
    """
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend)
    rank, world = _dist_info()
    _maybe_print(f"[imagenet_parallel] Initialized distributed backend={backend} | rank={rank}/{world}")


if __name__ == "__main__":
    """
    Quick smoke test: run with or without torch.distributed initialized.
    Note: this will iterate one batch per loader to show that data reads work.
    """
    # If you want to test distributed locally, wrap this file with torchrun:
    # torchrun --standalone --nproc_per_node=2 imagenet_parallel.py

    # Optionally initialize distributed if env vars are set by torchrun
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _maybe_print("[imagenet_parallel] torch.distributed already initialized")
    elif "LOCAL_RANK" in os.environ:
        init_distributed(int(os.environ["LOCAL_RANK"]))
    else:
        _maybe_print("[imagenet_parallel] Running in non-distributed mode", rank_only=False)

    rank, world = _dist_info()
    _maybe_print(f"[imagenet_parallel] Rank {rank} online out of {world}", rank_only=False)

    train_loader, val_loader, test_loader, samplers, _ = build_imagenet_distributed_loaders()

    # Pull a single batch from each loader to verify shape/labels
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        for images, labels in loader:
            _maybe_print(
                f"[imagenet_parallel] First {name} batch | imgs={tuple(images.shape)} | labels={tuple(labels.shape)}"
            )
            break

    _maybe_print("[imagenet_parallel] Smoke test complete")
