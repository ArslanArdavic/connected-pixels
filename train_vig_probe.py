# train_vig_probe.py
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

# Optional Neptune (same spirit as train_vig.py)
try:
    import neptune
except Exception:
    neptune = None

from vig_pytorch.vig import vig_ti_224_gelu

from imagenet_data import build_imagenet_loaders
from imagenet_data_masked import build_imagenet_masked_loaders


# -----------------------
# Basic configuration (edit these like train_vig.py)
# -----------------------
USE_NEPTUNE = True
NEPTUNE_PROJECT = "ALLab-Boun/connected-pixels"
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYjlhZjMxMS1mZjgyLTQ4Y2YtYmY5ZC1mMjVjOWU2YmI4YWMifQ==" # <-- put your token here

NW = 8

# Pretrain (MPP)
PRETRAIN_EPOCHS = 1
PRETRAIN_BS = 512
PRETRAIN_VAL_BS = 512
PRETRAIN_LR = 1e-5
PATCH_SIZE = 16
MASK_RATIO = 0.5

# Probe
PROBE_EPOCHS = 5
PROBE_TRAIN_BS = 512
PROBE_TEST_BS = 512
PROBE_LR = 1e-5

# Data split
VAL_FRACTION = 0.1

# ImageNet paths (keep same defaults as imagenet_data.py)
TRAIN_ROOT = "/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
VAL_ROOT = "/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
DEVKIT_ROOT = os.path.expanduser("~/connected-pixels/ILSVRC2012_devkit_t12")


# -----------------------
# Metrics
# -----------------------
def accuracy_topk(logits, targets, topk=(1, 5)):
    """Compute Top-k accuracies. logits: [B,C], targets: [B]."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B,maxk]
        pred = pred.t()  # [maxk,B]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [maxk,B]
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            res.append(correct_k / targets.size(0))
        return res


# -----------------------
# Token probing & hooks for ViG
# -----------------------
def _standardize_tokens(x, expected_num_patches):
    """
    Convert candidate tensor to tokens [B,N,D].
    Supports:
      - [B,N,D]
      - [B,D,N]
      - [B,C,H,W] where H*W == N  (treat spatial locations as patches)
    Returns (tokens_bnd, layout, extra) where extra stores (C,H,W) for restore.
    """
    if not torch.is_tensor(x):
        return None, None, None

    if x.dim() == 3:
        b, a, c = x.shape
        if a == expected_num_patches:
            return x, "BND", None
        if c == expected_num_patches:
            return x.transpose(1, 2).contiguous(), "BDN", None
        # fallback: treat as [B,N,D]
        return x, "BND", None

    if x.dim() == 4:
        b, c, h, w = x.shape
        if h * w != expected_num_patches:
            return None, None, None
        # [B,C,H,W] -> [B, H*W, C]
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens, "BCHW", (c, h, w)

    return None, None, None


def _restore_tokens(tokens_bnd, layout, extra):
    """Restore tokens back to original tensor layout."""
    if layout == "BND":
        return tokens_bnd
    if layout == "BDN":
        return tokens_bnd.transpose(1, 2).contiguous()
    if layout == "BCHW":
        c, h, w = extra
        # [B, H*W, C] -> [B,C,H,W]
        return tokens_bnd.transpose(1, 2).contiguous().view(tokens_bnd.size(0), c, h, w)
    return tokens_bnd


def _extract_tensors(obj):
    """Recursively pull tensors out of nested outputs."""
    out = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_extract_tensors(v))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_extract_tensors(v))
    return out


def find_token_modules(model, sample_images, expected_num_patches):
    """
    Dry-run forward once and find:
      - first module output that can be standardized to [B,N,D]
      - last  module output that can be standardized to [B,N,D]
    We allow 3D tokens or 4D feature maps with H*W == N, and we also look
    inside tuples/lists/dicts.
    """
    token_hits = []
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            tensors = _extract_tensors(out)
            for t in tensors:
                tok, layout, extra = _standardize_tokens(t, expected_num_patches)
                if tok is None:
                    continue
                token_hits.append((name, module, t.shape, layout, extra))
        return hook_fn

    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(sample_images)

    for h in hooks:
        h.remove()

    if len(token_hits) == 0:
        # Helpful debug: show some common tensor shapes seen (optional quick check)
        raise RuntimeError(
            "Could not find any token-like outputs matching N patches. "
            "This ViG likely does not expose a patch grid at 224/patch_size. "
            "Try printing model internals or using a built-in forward_features() if available."
        )

    patch_mod = token_hits[0]
    final_mod = token_hits[-1]
    return patch_mod, final_mod, token_hits
class ViGForMPP(nn.Module):
    """
    Wraps a ViG classifier and:
      1) corrupts patch-like tokens inside the forward pass via a hook (80/10/10 on 50% mask)
      2) captures final token-like features via another hook
      3) predicts mean 3-bit patch color targets with an MPP head

    Requires global helpers:
      - _standardize_tokens(x, expected_num_patches) -> (tokens_bnd, layout, extra)
      - _restore_tokens(tokens_bnd, layout, extra) -> original layout tensor
    """

    def __init__(self, base_model, expected_num_patches):
        super().__init__()
        self.base = base_model
        self.expected_num_patches = expected_num_patches

        # Hook targets
        self._patch_module = None
        self._final_module = None
        self._patch_hook = None
        self._final_hook = None

        # Per-forward state
        self._cur_mask = None          # [B,N] bool
        self._last_tokens = None       # [B,N,D] captured
        self._final_layout = None
        self._final_extra = None

        # Learnable mask token + MPP head (created in setup after we know token dim)
        self.mask_token = None         # nn.Parameter [1,1,D]
        self.mpp_head = None           # nn.Linear(D, 24)

    def remove_hooks(self):
        if self._patch_hook is not None:
            self._patch_hook.remove()
            self._patch_hook = None
        if self._final_hook is not None:
            self._final_hook.remove()
            self._final_hook = None

    def _corrupt_tokens_bnd(self, tokens_bnd):
        """
        tokens_bnd: [B,N,D]  -> returns corrupted tokens [B,N,D]
        Applies corruption only on self._cur_mask positions.
        """
        if self._cur_mask is None:
            return tokens_bnd

        B, N, D = tokens_bnd.shape
        mask = self._cur_mask
        if mask.device != tokens_bnd.device:
            mask = mask.to(tokens_bnd.device)

        if mask.shape != (B, N):
            return tokens_bnd

        r = torch.rand(B, N, device=tokens_bnd.device)
        do_mask = mask & (r < 0.80)
        do_rand = mask & (r >= 0.80) & (r < 0.90)
        # keep = mask & (r >= 0.90) -> unchanged

        flat = tokens_bnd.reshape(B * N, D)
        perm = torch.randperm(B * N, device=tokens_bnd.device)
        flat_shuf = flat[perm].reshape(B, N, D)

        out = tokens_bnd.clone()

        if do_mask.any():
            mtok = self.mask_token.expand(B, N, D)
            out[do_mask] = mtok[do_mask]

        if do_rand.any():
            out[do_rand] = flat_shuf[do_rand]

        return out

    def _replace_first_tokenlike_in_structure(self, obj):
        """
        Recursively finds the first tensor in `obj` that can be standardized to [B,N,D]
        (with N == expected_num_patches), corrupts it, restores it to original layout,
        and returns the updated structure.
        """
        # Tensor case
        if torch.is_tensor(obj):
            tokens_bnd, layout, extra = _standardize_tokens(obj, self.expected_num_patches)
            if tokens_bnd is None:
                return obj, False
            tokens_bnd = self._corrupt_tokens_bnd(tokens_bnd)
            restored = _restore_tokens(tokens_bnd, layout, extra)
            return restored, True

        # Tuple/list case
        if isinstance(obj, tuple):
            lst = list(obj)
            for i in range(len(lst)):
                new_v, done = self._replace_first_tokenlike_in_structure(lst[i])
                if done:
                    lst[i] = new_v
                    return tuple(lst), True
            return obj, False

        if isinstance(obj, list):
            lst = list(obj)
            for i in range(len(lst)):
                new_v, done = self._replace_first_tokenlike_in_structure(lst[i])
                if done:
                    lst[i] = new_v
                    return lst, True
            return obj, False

        # Dict case
        if isinstance(obj, dict):
            new_d = dict(obj)
            for k in new_d.keys():
                new_v, done = self._replace_first_tokenlike_in_structure(new_d[k])
                if done:
                    new_d[k] = new_v
                    return new_d, True
            return obj, False

        return obj, False

    def _capture_last_tokenlike_from_structure(self, obj):
        """
        Recursively scans `obj` and stores the last token-like tensor (standardized to [B,N,D])
        into self._last_tokens.
        """
        last = None
        last_layout = None
        last_extra = None

        def walk(x):
            nonlocal last, last_layout, last_extra
            if torch.is_tensor(x):
                tokens_bnd, layout, extra = _standardize_tokens(x, self.expected_num_patches)
                if tokens_bnd is not None:
                    last = tokens_bnd
                    last_layout = layout
                    last_extra = extra
                return
            if isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
                return
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
                return

        walk(obj)

        if last is not None:
            self._last_tokens = last
            self._final_layout = last_layout
            self._final_extra = last_extra

    def _corrupt_hook(self, module, inp, out):
        """
        Hook on the patch-embedding-like module output. We corrupt the first token-like
        tensor found in the module output structure and return the modified output.
        """
        if self._cur_mask is None:
            return out

        new_out, _done = self._replace_first_tokenlike_in_structure(out)
        return new_out

    def _capture_final_hook(self, module, inp, out):
        """
        Hook on a late module output. We capture the last token-like tensor found in out.
        """
        self._capture_last_tokenlike_from_structure(out)
        return out

    def setup(self, sample_images):
        """
        Discovers token-like patch module and final token module (using your find_token_modules),
        then installs hooks and initializes mask_token + mpp_head after inferring token dim.
        """
        # Discover modules
        patch_mod, final_mod, _hits = find_token_modules(self.base, sample_images, self.expected_num_patches)
        patch_name, patch_module, patch_shape, patch_layout, patch_extra = patch_mod
        final_name, final_module, final_shape, final_layout, final_extra = final_mod

        self._patch_module = patch_module
        self._final_module = final_module

        # Infer token dim from final module output
        tmp = {"tokens": None}

        def tmp_hook(_m, _i, o):
            # Capture last token-like tensor from o
            self._last_tokens = None
            self._capture_last_tokenlike_from_structure(o)
            if self._last_tokens is not None:
                tmp["tokens"] = self._last_tokens

        h = final_module.register_forward_hook(tmp_hook)
        self.base.eval()
        with torch.no_grad():
            _ = self.base(sample_images)
        h.remove()

        if tmp["tokens"] is None:
            raise RuntimeError("Failed to infer token dimension from final token-like module output.")

        tok_dim = tmp["tokens"].shape[-1]

        # Create learnable mask token + MPP head
        dev = sample_images.device
        self.mask_token = nn.Parameter(torch.zeros(1, 1, tok_dim, device=dev))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        self.mpp_head = nn.Linear(tok_dim, 24).to(dev)

        # Install hooks
        self.remove_hooks()
        self._patch_hook = patch_module.register_forward_hook(lambda m, i, o: self._corrupt_hook(m, i, o))
        self._final_hook = final_module.register_forward_hook(lambda m, i, o: self._capture_final_hook(m, i, o))

        print(f"[MPP] Patch token-like module: {patch_name} | example tensor shape: {patch_shape} | layout: {patch_layout}")
        print(f"[MPP] Final token-like module: {final_name} | example tensor shape: {final_shape} | layout: {final_layout}")
        print(f"[MPP] Inferred token dim: {tok_dim}")

    def forward(self, images, patch_mask):
        """
        Returns:
          base_logits: [B,1000] from original ViG head (kept intact)
          mpp_logits:  [B,N,3,8]
          tokens:      [B,N,D] final token-like features
        """
        if self.mask_token is None or self.mpp_head is None:
            raise RuntimeError("Call model.setup(sample_images) before using ViGForMPP.")

        self._cur_mask = patch_mask
        self._last_tokens = None
        self._final_layout = None
        self._final_extra = None

        base_logits = self.base(images)  # triggers corruption + capture hooks

        if self._last_tokens is None:
            raise RuntimeError(
                "Final token-like features were not captured. "
                "Hook discovery may have picked an incompatible module output."
            )

        B, N, D = self._last_tokens.shape
        out = self.mpp_head(self._last_tokens)   # [B,N,24]
        out = out.view(B, N, 3, 8)               # [B,N,3,8]
        return base_logits, out, self._last_tokens

def mpp_loss(mpp_logits, targets, patch_mask):
    """
    mpp_logits: [B,N,3,8]
    targets:    [B,N,3] in [0..7]
    patch_mask: [B,N] bool (loss only on masked patches)
    """
    B, N, C, K = mpp_logits.shape
    mask = patch_mask
    if mask.device != mpp_logits.device:
        mask = mask.to(mpp_logits.device)
    if targets.device != mpp_logits.device:
        targets = targets.to(mpp_logits.device)

    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=mpp_logits.device, requires_grad=True)

    # Select masked positions
    logits_m = mpp_logits[mask]   # [M,3,8]
    targ_m = targets[mask]        # [M,3]

    # CE per channel
    loss = 0.0
    for ch in range(3):
        loss = loss + nn.functional.cross_entropy(logits_m[:, ch, :], targ_m[:, ch])
    return loss / 3.0


# -----------------------
# Train / Eval loops
# -----------------------
def train_one_epoch_mpp(model, loader, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0

    num_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx % 1000 == 0:
            print(f"[MPP-TRAIN] Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{num_batches}")

        if len(batch) == 3:
            images, patch_mask, targets = batch
        else:
            images, patch_mask, targets, _cls = batch

        images = images.to(device, non_blocking=True)
        patch_mask = patch_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        _base_logits, mpp_logits, _tokens = model(images, patch_mask)
        loss = mpp_loss(mpp_logits, targets, patch_mask)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    print(f"[MPP-TRAIN] Epoch {epoch}/{num_epochs} - Done. Loss: {avg_loss:.6f}")
    return avg_loss


def eval_one_epoch_mpp(model, loader, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        num_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            if batch_idx % 1000 == 0:
                print(f"[MPP-VAL] Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{num_batches}")

            if len(batch) == 3:
                images, patch_mask, targets = batch
            else:
                images, patch_mask, targets, _cls = batch

            images = images.to(device, non_blocking=True)
            patch_mask = patch_mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            _base_logits, mpp_logits, _tokens = model(images, patch_mask)
            loss = mpp_loss(mpp_logits, targets, patch_mask)

            bs = images.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    print(f"[MPP-VAL] Epoch {epoch}/{num_epochs} - Done. Loss: {avg_loss:.6f}")
    return avg_loss


def set_linear_probe_trainable(model):
    """
    Freeze everything except the final classifier head.
    We try common attribute names; if not found, we fall back to name matching.
    """
    for p in model.parameters():
        p.requires_grad = False

    head = None
    for attr in ["head", "classifier", "fc", "mlp_head"]:
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                head = mod
                break

    if head is not None:
        for p in head.parameters():
            p.requires_grad = True
        print(f"[PROBE] Unfroze classifier module: {attr}")
        return

    # Fallback: unfreeze parameters with typical head-like names
    unfrozen = 0
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ["head", "classifier", "fc", "mlp_head"]):
            p.requires_grad = True
            unfrozen += 1
    print(f"[PROBE] Fallback unfreeze by name-match. Unfroze {unfrozen} params.")
    if unfrozen == 0:
        raise RuntimeError("Could not identify classifier head parameters to train for linear probe.")


def train_one_epoch_probe(model, loader, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0
    sum_top1 = 0.0
    sum_top5 = 0.0

    num_batches = len(loader)
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        if batch_idx % 1000 == 0:
            print(f"[PROBE-TRAIN] Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{num_batches}")

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        top1, top5 = accuracy_topk(logits, labels, topk=(1, 5))
        sum_top1 += top1 * bs
        sum_top5 += top5 * bs

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    avg_top1 = sum_top1 / total_samples if total_samples > 0 else 0.0
    avg_top5 = sum_top5 / total_samples if total_samples > 0 else 0.0

    print(f"[PROBE-TRAIN] Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f} | Top1: {avg_top1:.4f} | Top5: {avg_top5:.4f}")
    return avg_loss, avg_top1, avg_top5


def eval_probe(model, loader, device, split_name="VAL"):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    sum_top1 = 0.0
    sum_top5 = 0.0

    with torch.no_grad():
        num_batches = len(loader)
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            print(f"[PROBE-{split_name}] Batch {batch_idx}/{num_batches}")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            bs = images.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

            top1, top5 = accuracy_topk(logits, labels, topk=(1, 5))
            sum_top1 += top1 * bs
            sum_top5 += top5 * bs

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    avg_top1 = sum_top1 / total_samples if total_samples > 0 else 0.0
    avg_top5 = sum_top5 / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_top1, avg_top5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join("saved_models", "vig_ti_mpp_probe", timestamp)
    os.makedirs(model_dir, exist_ok=True)

    mpp_ckpt_path = os.path.join(model_dir, "mpp_pretrained.pth")
    probe_ckpt_path = os.path.join(model_dir, "linear_probe_imagenet1k.pth")
    log_path = os.path.join(model_dir, "log.txt")

    log_lines = []
    log_lines.append(f"Device: {device}")
    log_lines.append(f"Timestamp: {timestamp}")
    log_lines.append("")
    log_lines.append("[MPP]")
    log_lines.append(f"  epochs: {PRETRAIN_EPOCHS}")
    log_lines.append(f"  batch_size: {PRETRAIN_BS}")
    log_lines.append(f"  lr: {PRETRAIN_LR}")
    log_lines.append(f"  patch_size: {PATCH_SIZE}")
    log_lines.append(f"  mask_ratio: {MASK_RATIO}")
    log_lines.append("")
    log_lines.append("[PROBE]")
    log_lines.append(f"  epochs: {PROBE_EPOCHS}")
    log_lines.append(f"  train_bs: {PROBE_TRAIN_BS}")
    log_lines.append(f"  test_bs: {PROBE_TEST_BS}")
    log_lines.append(f"  lr: {PROBE_LR}")
    log_lines.append("")

    # Neptune init (optional)
    run = None
    if USE_NEPTUNE and neptune is not None and NEPTUNE_API_TOKEN:
        tags = ["vig_ti", "mpp", "linear_probe", f"mpp_lr_{PRETRAIN_LR}", f"probe_lr_{PROBE_LR}"]
        run = neptune.init_run(
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_TOKEN,
            name="vig_ti mpp + linear_probe",
            tags=tags,
        )
        run["config"] = {
            "model": "vig_ti_224_gelu",
            "timestamp": timestamp,
            "mpp": {
                "epochs": PRETRAIN_EPOCHS,
                "batch_size": PRETRAIN_BS,
                "val_batch_size": PRETRAIN_VAL_BS,
                "lr": PRETRAIN_LR,
                "patch_size": PATCH_SIZE,
                "mask_ratio": MASK_RATIO,
                "val_fraction": VAL_FRACTION,
            },
            "probe": {
                "epochs": PROBE_EPOCHS,
                "train_bs": PROBE_TRAIN_BS,
                "test_bs": PROBE_TEST_BS,
                "lr": PROBE_LR,
                "val_fraction": VAL_FRACTION,
            },
        }

    start_time = time.time()

    # -----------------------
    # Stage A: MPP pretraining
    # -----------------------
    print("[STAGE A] Building masked pretraining loaders...")
    pretrain_train_loader, pretrain_val_loader, _wnid = build_imagenet_masked_loaders(
        train_root=TRAIN_ROOT,
        devkit_root=DEVKIT_ROOT,
        train_batch_size=PRETRAIN_BS,
        val_batch_size=PRETRAIN_VAL_BS,
        num_workers=NW,
        val_fraction=VAL_FRACTION,
        patch_size=PATCH_SIZE,
        mask_ratio=MASK_RATIO,
        return_label=False,
    )

    expected_num_patches = (224 // PATCH_SIZE) * (224 // PATCH_SIZE)

    print("[STAGE A] Initializing ViG + MPP wrapper...")
    base = vig_ti_224_gelu(num_classes=1000)  # default classifier exists; we keep it untouched
    base.to(device)

    mpp_model = ViGForMPP(base, expected_num_patches=expected_num_patches).to(device)

    # Discover token modules and create mask_token/head
    sample_images, sample_mask, sample_targets = next(iter(pretrain_train_loader))
    sample_images = sample_images.to(device, non_blocking=True)
    mpp_model.setup(sample_images)

    optimizer_mpp = torch.optim.Adam(mpp_model.parameters(), lr=PRETRAIN_LR)

    best_val_loss = 1e9
    print("[STAGE A] MPP training started.")
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        train_loss = train_one_epoch_mpp(mpp_model, pretrain_train_loader, optimizer_mpp, device, epoch, PRETRAIN_EPOCHS)
        val_loss = eval_one_epoch_mpp(mpp_model, pretrain_val_loader, device, epoch, PRETRAIN_EPOCHS)

        log_lines.append(f"[MPP] Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if run is not None:
            run["mpp/train_loss"].append(train_loss)
            run["mpp/val_loss"].append(val_loss)
            run["mpp/epoch"].log(epoch)

        # Save best MPP checkpoint (separate)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "stage": "mpp_pretrain",
                    "timestamp": timestamp,
                    "expected_num_patches": expected_num_patches,
                    "patch_size": PATCH_SIZE,
                    "mask_ratio": MASK_RATIO,
                    "base_state_dict": mpp_model.base.state_dict(),
                    "mask_token": mpp_model.mask_token.detach().cpu(),
                    "mpp_head_state_dict": mpp_model.mpp_head.state_dict(),
                    "optimizer_state_dict": optimizer_mpp.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                mpp_ckpt_path,
            )
            print(f"[STAGE A] Saved best MPP checkpoint to: {mpp_ckpt_path}")

            if run is not None:
                run["artifacts/mpp_pretrained"].upload(mpp_ckpt_path)

    print(f"[STAGE A] Done. Best val loss: {best_val_loss:.6f}")

    # -----------------------
    # Stage B: Linear probe
    # -----------------------
    print("[STAGE B] Building supervised loaders (for linear probe)...")
    train_loader, val_loader, test_loader, _wnid2 = build_imagenet_loaders(
        train_root=TRAIN_ROOT,
        val_root=VAL_ROOT,
        devkit_root=DEVKIT_ROOT,
        train_batch_size=PROBE_TRAIN_BS,
        test_batch_size=PROBE_TEST_BS,
        num_workers=NW,
        val_fraction=VAL_FRACTION,
    )

    print("[STAGE B] Initializing probe model...")
    probe_model = vig_ti_224_gelu(num_classes=1000)
    probe_model.to(device)

    # Load pretrained backbone weights from MPP checkpoint (separate file)
    ckpt = torch.load(mpp_ckpt_path, map_location="cpu")
    missing, unexpected = probe_model.load_state_dict(ckpt["base_state_dict"], strict=False)
    print(f"[STAGE B] Loaded MPP backbone weights. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    # Freeze backbone, train only head
    set_linear_probe_trainable(probe_model)

    # Optimizer over trainable params only
    trainable_params = [p for p in probe_model.parameters() if p.requires_grad]
    optimizer_probe = torch.optim.Adam(trainable_params, lr=PROBE_LR)

    best_val_top1 = -1.0
    print("[STAGE B] Linear probe training started.")
    for epoch in range(1, PROBE_EPOCHS + 1):
        tr_loss, tr_top1, tr_top5 = train_one_epoch_probe(probe_model, train_loader, optimizer_probe, device, epoch, PROBE_EPOCHS)
        va_loss, va_top1, va_top5 = eval_probe(probe_model, val_loader, device, split_name="VAL")

        print(f"[PROBE] Epoch {epoch}/{PROBE_EPOCHS} | Val Loss: {va_loss:.6f} | Val Top1: {va_top1:.4f} | Val Top5: {va_top5:.4f}")

        log_lines.append(
            f"[PROBE] Epoch {epoch}: train_loss={tr_loss:.6f} train_top1={tr_top1:.4f} train_top5={tr_top5:.4f} "
            f"val_loss={va_loss:.6f} val_top1={va_top1:.4f} val_top5={va_top5:.4f}"
        )

        if run is not None:
            run["probe/train_loss"].append(tr_loss)
            run["probe/train_top1"].append(tr_top1)
            run["probe/train_top5"].append(tr_top5)
            run["probe/val_loss"].append(va_loss)
            run["probe/val_top1"].append(va_top1)
            run["probe/val_top5"].append(va_top5)
            run["probe/epoch"].log(epoch)

        # Save best probe checkpoint (separate)
        if va_top1 > best_val_top1:
            best_val_top1 = va_top1
            torch.save(
                {
                    "stage": "linear_probe",
                    "timestamp": timestamp,
                    "base_from_mpp": mpp_ckpt_path,
                    "model_state_dict": probe_model.state_dict(),
                    "optimizer_state_dict": optimizer_probe.state_dict(),
                    "epoch": epoch,
                    "best_val_top1": best_val_top1,
                },
                probe_ckpt_path,
            )
            print(f"[STAGE B] Saved best probe checkpoint to: {probe_ckpt_path}")

            if run is not None:
                run["artifacts/linear_probe"].upload(probe_ckpt_path)

    # Final test
    te_loss, te_top1, te_top5 = eval_probe(probe_model, test_loader, device, split_name="TEST")
    print(f"[TEST] Loss: {te_loss:.6f} | Top1: {te_top1:.4f} | Top5: {te_top5:.4f}")

    log_lines.append("")
    log_lines.append("[TEST RESULTS]")
    log_lines.append(f"  test_loss: {te_loss:.6f}")
    log_lines.append(f"  test_top1: {te_top1:.6f}")
    log_lines.append(f"  test_top5: {te_top5:.6f}")

    if run is not None:
        run["test/loss"] = te_loss
        run["test/top1"] = te_top1
        run["test/top5"] = te_top5

    total_time_sec = time.time() - start_time
    log_lines.append("")
    log_lines.append(f"Total time (s): {total_time_sec:.2f}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"Wrote log to: {log_path}")

    if run is not None:
        run["time/total_sec"] = total_time_sec
        run["artifacts/log"].upload(log_path)
        # Also ensure final artifacts are uploaded (in case “best” never triggered)
        run["artifacts/mpp_pretrained"].upload(mpp_ckpt_path)
        run["artifacts/linear_probe"].upload(probe_ckpt_path)
        run.stop()


if __name__ == "__main__":
    main()
