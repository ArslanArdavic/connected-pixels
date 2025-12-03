import os
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16

from sklearn.metrics import f1_score

from imagenet_data import build_imagenet_loaders  # your loaders

# -----------------------
# Neptune setup
# -----------------------
USE_NEPTUNE = True  # set False if you want to disable logging
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYjlhZjMxMS1mZjgyLTQ4Y2YtYmY5ZC1mMjVjOWU2YmI4YWMifQ==" # <-- put your token here

try:
    import neptune
except ImportError:
    neptune = None
    USE_NEPTUNE = False
    print("[WARN] Neptune is not installed; disabling Neptune logging.")


def parse_args():
    parser = argparse.ArgumentParser(description="ViT-B/16 ImageNet training")

    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    # optional sweep meta-info
    parser.add_argument("--tag", action="append", default=None,
                        help="Additional Neptune tags (can be used multiple times)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional custom run name for Neptune")

    return parser.parse_args()


def compute_accuracy(logits, targets):
    """Top-1 accuracy."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total


def evaluate(model, data_loader, device):
    """Evaluate accuracy on a given loader."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        num_batches = len(data_loader)
        for batch_idx, (images, labels) in enumerate(data_loader, start=1):
            print(f"[VAL] Batch {batch_idx}/{num_batches}")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            correct, total = compute_accuracy(outputs, labels)
            total_correct += correct
            total_samples += total

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return acc


def test_metrics(model, data_loader, device):
    """Compute test accuracy and macro F1-score."""
    model.eval()
    all_targets = []
    all_preds = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        num_batches = len(data_loader)
        for batch_idx, (images, labels) in enumerate(data_loader, start=1):
            print(f"[TEST] Batch {batch_idx}/{num_batches}")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro")  # macro-F1 for multi-class
    return acc, f1


def main():
    args = parse_args()

    # -----------------------
    # Basic configuration
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_workers = args.num_workers
    num_epochs = args.epochs
    lr = args.lr

    # Make sure directories exist
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # -----------------------
    # Data loaders
    # -----------------------
    train_loader, val_loader, test_loader, wnid_to_idx = build_imagenet_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )

    num_classes = 1000  # ImageNet-1K

    # -----------------------
    # Model, loss, optimizer
    # -----------------------
    model = vit_b_16(weights=None)  # no pretraining, pure classification head
    # Ensure classifier head matches num_classes (should be 1000 by default, but explicit is safer)
    if model.heads.head.out_features != num_classes:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------
    # Name + logging setup
    # -----------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"vit_b_16_epoch{num_epochs}_lr{lr}_bs{train_batch_size}_{timestamp}"
    model_name = args.run_name if args.run_name is not None else base_name
    model_path = os.path.join("saved_models", model_name + ".pth")
    log_path = os.path.join("outputs", model_name + ".txt")

    log_lines = []
    log_lines.append(f"Model name: {model_name}")
    log_lines.append(f"Device: {device}")
    log_lines.append(f"Epochs: {num_epochs}")
    log_lines.append(f"Train batch size: {train_batch_size}")
    log_lines.append(f"Test batch size: {test_batch_size}")
    log_lines.append(f"Num workers: {num_workers}")
    log_lines.append(f"Optimizer: Adam")
    log_lines.append(f"Learning rate: {lr}")
    log_lines.append(f"Num classes: {num_classes}")
    log_lines.append("")

    # -----------------------
    # Neptune run init
    # -----------------------
    run = None
    if USE_NEPTUNE and neptune is not None:
        extra_tags = args.tag if args.tag is not None else []
        tags = ["vit_b_16", "supervised", "sweep"] + extra_tags

        run = neptune.init_run(
            project="ALLab-Boun/connected-pixels",
            api_token=NEPTUNE_API_TOKEN,
            name=model_name,
            tags=tags,
        )

        # Log configuration
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
        }

    # -----------------------
    # Training (epochs)
    # -----------------------
    print("Training started.")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            # Show which batch is currently processed
            if batch_idx % 1000 == 0:
                print(f"[TRAIN] Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{num_batches}")

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
        
        
        print(f"[TRAIN] Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{num_batches}")

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        # -----------------------
        # Validation after epoch
        # -----------------------
        val_acc = evaluate(model, val_loader, device)

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

        # Log to Neptune
        if run is not None:
            run["train/loss"].append(epoch_loss)
            run["train/acc"].append(train_acc)
            run["val/acc"].append(val_acc)
            run["epoch"].log(epoch)

    # -----------------------
    # Test evaluation
    # -----------------------
    test_acc, test_f1 = test_metrics(model, test_loader, device)
    print(f"[TEST] Accuracy: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    log_lines.append("Test results:")
    log_lines.append(f"  Test Acc: {test_acc:.6f}")
    log_lines.append(f"  Test Macro F1: {test_f1:.6f}")
    log_lines.append("")

    # Log test metrics to Neptune
    if run is not None:
        run["test/acc"] = test_acc
        run["test/macro_f1"] = test_f1

    # -----------------------
    # Time & saving
    # -----------------------
    total_time_sec = time.time() - start_time
    log_lines.append(f"Total training + eval time (s): {total_time_sec:.2f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # Write log file
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"Wrote log to: {log_path}")

    # Upload artifacts to Neptune
    if run is not None:
        run["time/total_sec"] = total_time_sec
        run["artifacts/model"].upload(model_path)
        run["artifacts/log"].upload(log_path)
        run.stop()


if __name__ == "__main__":
    main()