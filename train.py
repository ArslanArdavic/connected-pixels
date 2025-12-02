import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from sklearn.metrics import f1_score

import neptune
import tqdm

from imagenet_data import build_imagenet_loaders 


NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYjlhZjMxMS1mZjgyLTQ4Y2YtYmY5ZC1mMjVjOWU2YmI4YWMifQ==" 

def parse_args():
    parser = argparse.ArgumentParser(description="ViT-B/16 ImageNet training")

    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)

    parser.add_argument("--tag", action="append", default=None,
                        help="Neptune tags")
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
        total_batch = len(data_loader)
        for batch_idx, (images, labels) in enumerate(data_loader, start=1):
            if batch_idx % 50 == 0:
                print(f"[VAL] Batch ({batch_idx}/{total_batch})")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            correct, total = compute_accuracy(outputs, labels)
            total_correct += correct
            total_samples += total
        
        print(f"[VAL] Batch ({batch_idx}/{total_batch})")

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
        total_batch = len(data_loader)
        for batch_idx, (images, labels) in enumerate(data_loader, start=1):
            if batch_idx % 50 == 0:
                print(f"[TEST] Batch ({batch_idx}/{total_batch})")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
        
        print(f"[TEST] Batch ({batch_idx}/{total_batch})")

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro")  # macro-F1 for multi-class
    return acc, f1

def main():
    
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_workers = args.num_workers
    num_epochs = args.epochs
    lr = args.lr
    
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_loader, val_loader, test_loader, wnid_to_idx = build_imagenet_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )

    num_classes = 1000 

    model = vit_b_16(weights=None)     
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=lr)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    extra_tags = args.tag if args.tag is not None else []
    tags = ["vit_b_16", "classification"] + extra_tags

    run = neptune.init_run(
        project="ALLab-Boun/connected-pixels",
        api_token=NEPTUNE_API_TOKEN,
        name=args.run_name if args.run_name is not None else "vit_b_16 base config",
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
        }
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        total_batch = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            if batch_idx % 50 == 0:
                print(f"[TRAIN] Epoch ({epoch}/ {num_epochs}) -- Batch ({batch_idx}/{total_batch})")

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
        
        
        print(f"[TRAIN] Epoch ({epoch}/ {num_epochs}) -- Batch ({batch_idx}/{total_batch})")
        
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        val_acc = evaluate(model, val_loader, device)

        run["train/loss"].append(epoch_loss)
        run["train/acc"].append(train_acc)
        run["val/acc"].append(val_acc)
        run["epoch"].log(epoch)