import os
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import vit_b_16
from sklearn.metrics import f1_score

# -----------------------
# Neptune setup (same as train_neptune.py)
# -----------------------
USE_NEPTUNE = True  # set False if you want to disable logging
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYjlhZjMxMS1mZjgyLTQ4Y2YtYmY5ZC1mMjVjOWU2YmI4YWMifQ=="  # <-- put your token here

try:
    import neptune
except ImportError:
    neptune = None
    USE_NEPTUNE = False
    print("[WARN] Neptune is not installed; disabling Neptune logging.")

from imagenet_parallel import (
    build_imagenet_distributed_loaders,
    set_epoch_for_samplers,
    init_distributed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ViT-B/16 ImageNet training (multi-GPU DDP)")
    parser.add_argument("--train-batch-size", type=int, default=256, help="Per-GPU batch size")
    parser.add_argument("--test-batch-size", type=int, default=256, help="Per-GPU batch size for val/test")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by torchrun")
    # optional sweep meta-info (same as train_neptune.py)
    parser.add_argument("--tag", action="append", default=None, help="Additional Neptune tags (can be used multiple times)")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom run name for Neptune")
    return parser.parse_args()


def get_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def reduce_sum(value_tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    return value_tensor


def compute_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total


def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        num_batches = len(data_loader)
        for batch_idx, (images, labels) in enumerate(data_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            correct, total = compute_accuracy(outputs, labels)
            total_correct += correct
            total_samples += total

    # Reduce across ranks
    correct_tensor = torch.tensor(total_correct, device=device, dtype=torch.float64)
    total_tensor = torch.tensor(total_samples, device=device, dtype=torch.float64)
    correct_tensor = reduce_sum(correct_tensor)
    total_tensor = reduce_sum(total_tensor)

    acc = (correct_tensor / total_tensor).item() if total_tensor.item() > 0 else 0.0
    return acc


def test_metrics(model, data_loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        num_batches = len(data_loader)
        for batch_idx, (images, labels) in enumerate(data_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())

    # Gather across ranks for metrics
    if dist.is_available() and dist.is_initialized():
        # Gather scalars
        correct_tensor = torch.tensor(total_correct, device=device, dtype=torch.float64)
        total_tensor = torch.tensor(total_samples, device=device, dtype=torch.float64)
        correct_tensor = reduce_sum(correct_tensor)
        total_tensor = reduce_sum(total_tensor)

        # Gather targets/preds using all_gather_object to handle variable last batch sizes
        gathered_targets = [None for _ in range(dist.get_world_size())]
        gathered_preds = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_targets, torch.cat(all_targets) if all_targets else torch.empty(0))
        dist.all_gather_object(gathered_preds, torch.cat(all_preds) if all_preds else torch.empty(0))
        if dist.get_rank() == 0:
            all_targets = torch.cat(gathered_targets).numpy()
            all_preds = torch.cat(gathered_preds).numpy()
            acc = (correct_tensor / total_tensor).item() if total_tensor.item() > 0 else 0.0
            f1 = f1_score(all_targets, all_preds, average="macro")
            return acc, f1
        else:
            return None, None
    else:
        all_targets = torch.cat(all_targets).numpy() if all_targets else []
        all_preds = torch.cat(all_preds).numpy() if all_preds else []
        acc = total_correct / total_samples if total_samples > 0 else 0.0
        f1 = f1_score(all_targets, all_preds, average="macro") if len(all_targets) > 0 else 0.0
        return acc, f1


def main():
    args = parse_args()

    # Distributed init (torchrun sets LOCAL_RANK/RANK/WORLD_SIZE)
    if dist.is_available() and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        init_distributed(local_rank)

    rank, world_size = get_rank_world()

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_workers = args.num_workers
    num_epochs = args.epochs
    lr = args.lr

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_loader, val_loader, test_loader, samplers, wnid_to_idx = build_imagenet_distributed_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        distributed=True,
    )

    num_classes = 1000
    model = vit_b_16(weights=None)
    if model.heads.head.out_features != num_classes:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    model.to(device)
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[device], output_device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join("saved_models", "vit_b_16_parallel", timestamp)
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    log_path = os.path.join(model_dir, "log.txt")

    log_lines = []
    if rank == 0:
        log_lines.append(f"World size: {world_size}")
        log_lines.append(f"Per-GPU train batch size: {train_batch_size}")
        log_lines.append(f"Per-GPU val/test batch size: {test_batch_size}")
        log_lines.append(f"Effective global train batch: {train_batch_size * world_size}")
        log_lines.append(f"Num workers: {num_workers}")
        log_lines.append(f"Learning rate: {lr}")
        log_lines.append("")

    # Neptune run init (rank 0 only)
    run = None
    if rank == 0 and USE_NEPTUNE and neptune is not None:
        extra_tags = args.tag if args.tag is not None else []
        tags = ["vit_b_16",  "gpu_4", "batch_256_per_gpu","lr_2e-5"] + extra_tags

        run_name = args.run_name if args.run_name is not None else "vit_b_16 parallel multi_gpu"
        run = neptune.init_run(
            project="ALLab-Boun/connected-pixels",
            api_token=NEPTUNE_API_TOKEN,
            name=run_name,
            tags=tags,
        )

        run["config"] = {
            "model": "vit_b_16",
            "epochs": num_epochs,
            "train_batch_size": train_batch_size,
            "test_batch_size": test_batch_size,
            "num_workers": num_workers,
            "optimizer": "Adam",
            "lr": lr,
            "num_classes": num_classes,
            "timestamp": timestamp,
            "world_size": world_size,
        }

    if rank == 0:
        print("Training started.")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        set_epoch_for_samplers(epoch, samplers)

        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct, total = compute_accuracy(outputs, labels)
            total_correct += correct
            total_samples += total

            if rank == 0 and batch_idx % 100 == 0:
                print(f"[TRAIN] Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{num_batches}")

        # Reduce metrics across ranks
        loss_tensor = torch.tensor(running_loss, device=device, dtype=torch.float64)
        correct_tensor = torch.tensor(total_correct, device=device, dtype=torch.float64)
        total_tensor = torch.tensor(total_samples, device=device, dtype=torch.float64)
        loss_tensor = reduce_sum(loss_tensor)
        correct_tensor = reduce_sum(correct_tensor)
        total_tensor = reduce_sum(total_tensor)

        epoch_loss = (loss_tensor / total_tensor).item() if total_tensor.item() > 0 else 0.0
        train_acc = (correct_tensor / total_tensor).item() if total_tensor.item() > 0 else 0.0

        val_acc = evaluate(model, val_loader, device)

        if rank == 0:
            print(
                f"[EPOCH {epoch}] "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            log_lines.append(f"Epoch {epoch}:")
            log_lines.append(f"  Train Loss: {epoch_loss:.6f}")
            log_lines.append(f"  Train Acc:  {train_acc:.6f}")
            log_lines.append(f"  Val Acc:    {val_acc:.6f}")
            log_lines.append("")

            if run is not None:
                run["train/loss"].append(epoch_loss)
                run["train/acc"].append(train_acc)
                run["val/acc"].append(val_acc)
                run["epoch"].log(epoch)

    test_acc, test_f1 = test_metrics(model, test_loader, device)
    if rank == 0:
        print(f"[TEST] Accuracy: {test_acc:.4f} | Macro F1: {test_f1:.4f}")
        log_lines.append("Test results:")
        log_lines.append(f"  Test Acc: {test_acc:.6f}")
        log_lines.append(f"  Test Macro F1: {test_f1:.6f}")
        log_lines.append("")

        if run is not None:
            run["test/acc"] = test_acc
            run["test/macro_f1"] = test_f1

    total_time_sec = time.time() - start_time
    if rank == 0:
        log_lines.append(f"Total training + eval time (s): {total_time_sec:.2f}")

        torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), model_path)
        print(f"Saved model to: {model_path}")

        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))
        print(f"Wrote log to: {log_path}")

        if run is not None:
            run["time/total_sec"] = total_time_sec
            run["artifacts/model"].upload(model_path)
            run["artifacts/log"].upload(log_path)
            run.stop()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
