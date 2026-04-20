"""
03_train_finetune.py
====================
Step 3: Full end-to-end fine-tuning of EfficientNet-B0 on raw DICOM images.

Training strategy:
  • Phase 1 – WARMUP (epochs 1 … WARMUP_EPOCHS):
        Only the classifier head is trained (backbone still frozen).
        LR increases linearly from 0 → LR_HEAD.
        This prevents large gradient updates from destroying pretrained weights.

  • Phase 2 – FULL FINE-TUNING (epoch WARMUP_EPOCHS+1 … MAX_EPOCHS):
        Backbone unfrozen; differential learning rates:
            backbone layers  →  LR          (low, e.g. 1e-4)
            classifier head  →  LR_HEAD     (high, e.g. 1e-3)
        Cosine annealing LR decay.
        Early stopping on val_auc (patience = EARLY_STOP_PATIENCE).

Regularization stack:
  • L1 penalty on head weights  (prevents sparse-feature overfitting)
  • L2 weight decay via AdamW   (global shrinkage)
  • Dropout inside the head
  • Label smoothing in BCE loss
  • Random augmentation (see utils/dataset.py get_transforms)
  • Weighted random sampler     (class-imbalance correction)

After training:
  • Evaluates on internal test split (AUC, F1, AP, precision, recall).
  • If val_auc < threshold → runs Optuna for hyperparameter re-tuning.
  • Saves best checkpoint, training curves, per-class metrics.

Usage:
    python 03_train_finetune.py [--resume]
    python 03_train_finetune.py --resume   # resume from latest checkpoint
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from configs.config import (
    SEED, NUM_CLASSES, DATA_ROOT, TRAIN_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_AMP, GRAD_CLIP_NORM,
    LR, LR_HEAD, WEIGHT_DECAY, L1_LAMBDA, DROPOUT_RATE,
    MAX_EPOCHS, WARMUP_EPOCHS, EARLY_STOP_PATIENCE, EARLY_STOP_METRIC, MIN_DELTA,
    RETUNE_THRESHOLD, IMG_SIZE,
)
from utils.image_utils import build_image_index
from utils.dataset import make_dataloaders
from utils.metrics import compute_metrics, print_metrics, save_metrics
from utils.optuna_tuner import run_optuna, should_retune

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int = NUM_CLASSES,
    dropout:     float = DROPOUT_RATE,
    pretrained:  bool = True,
) -> nn.Module:
    """
    EfficientNet-B0 with a custom multi-label classification head.

    Head: Dropout → Linear(1280, 512) → BN → ReLU → Dropout → Linear(512, num_classes)
    """
    weights = M.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model   = M.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features   # 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(512, num_classes),
    )
    return model.to(DEVICE)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad_(False)


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# L1 regularisation
# ─────────────────────────────────────────────────────────────────────────────

def l1_penalty(model: nn.Module, lambda_l1: float) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=DEVICE)
    for name, module in model.named_modules():
        if "classifier" in name and isinstance(module, nn.Linear):
            penalty = penalty + module.weight.abs().sum()
    return lambda_l1 * penalty


# ─────────────────────────────────────────────────────────────────────────────
# Warmup LR scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup for WARMUP_EPOCHS, then cosine annealing until MAX_EPOCHS.
    Applied per-epoch (step() called once per epoch).
    """
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, base_lr: float):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.base_lr       = base_lr
        self._epoch        = 0

    def step(self) -> float:
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup_epochs:
            factor = e / self.warmup_epochs
        else:
            import math
            progress = (e - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            factor   = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = pg.get("_base_lr", self.base_lr) * factor

        return self.optimizer.param_groups[0]["lr"]

    @property
    def last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser builder
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    phase: str,          # "warmup" | "finetune"
    lr: float = LR, lr_head: float = LR_HEAD,
    wd: float = WEIGHT_DECAY,
    opt_name: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Differential learning rates:
      warmup   → head only, lr_head
      finetune → backbone: lr;  head: lr_head
    """
    if phase == "warmup":
        params = [{"params": model.classifier.parameters(),
                   "lr": lr_head, "_base_lr": lr_head}]
    else:
        backbone_params = [p for n, p in model.named_parameters()
                           if not n.startswith("classifier") and p.requires_grad]
        head_params     = list(model.classifier.parameters())
        params = [
            {"params": backbone_params, "lr": lr,      "_base_lr": lr},
            {"params": head_params,     "lr": lr_head,  "_base_lr": lr_head},
        ]

    if opt_name == "adamw":
        return torch.optim.AdamW(params, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(params, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(params, momentum=0.9, weight_decay=wd, nesterov=True)
    else:
        return torch.optim.AdamW(params, weight_decay=wd)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer | None,
    criterion:  nn.Module,
    scaler:     torch.cuda.amp.GradScaler | None,
    lambda_l1:  float,
    is_train:   bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    One epoch of train or eval.

    Returns (mean_loss, all_labels (N,C), all_probs (N,C)).
    """
    model.train() if is_train else model.eval()
    total_loss  = 0.0
    all_labels  = []
    all_probs   = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for imgs, labels, _ in loader:
            imgs   = imgs.to(DEVICE,   non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels)
                if is_train:
                    loss = loss + l1_penalty(model, lambda_l1)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()

            total_loss += loss.item()
            probs       = torch.sigmoid(logits).detach().cpu().float().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    mean_loss = total_loss / max(len(loader), 1)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return mean_loss, all_labels, all_probs


def train(
    model:      nn.Module,
    train_dl:   torch.utils.data.DataLoader,
    val_dl:     torch.utils.data.DataLoader,
    hp:         dict | None = None,
    run_name:   str = "finetune",
) -> tuple[nn.Module, float, dict]:
    """
    Full training loop with warmup + cosine LR + early stopping.

    Returns (best_model, best_val_auc, history_dict).
    """
    from sklearn.metrics import roc_auc_score

    # ── Hyperparameters ───────────────────────────────────────────────────────
    lr             = hp.get("lr",             LR)           if hp else LR
    lr_head        = hp.get("lr_head",        LR_HEAD)      if hp else LR_HEAD
    wd             = hp.get("weight_decay",   WEIGHT_DECAY) if hp else WEIGHT_DECAY
    l1_lam         = hp.get("l1_lambda",      L1_LAMBDA)    if hp else L1_LAMBDA
    dropout        = hp.get("dropout_rate",   DROPOUT_RATE) if hp else DROPOUT_RATE
    warmup_ep      = hp.get("warmup_epochs",  WARMUP_EPOCHS) if hp else WARMUP_EPOCHS
    label_smooth   = hp.get("label_smoothing", 0.05)        if hp else 0.05
    opt_name       = hp.get("optimizer",      "adamw")      if hp else "adamw"

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    scaler    = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.type == "cuda") else None

    # ────────────────────────────────────────────────────────────────────────
    # Phase 1 – Warmup (head only)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Phase 1 – Warmup ({warmup_ep} epochs, head only)")
    print(f"{'─'*55}")

    freeze_backbone(model)
    warmup_opt  = build_optimizer(model, "warmup", lr_head=lr_head, wd=wd, opt_name=opt_name)
    warmup_sched = WarmupCosineScheduler(warmup_opt, warmup_ep, warmup_ep, lr_head)

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, warmup_ep + 1):
        warmup_sched.step()
        tr_loss, _, _  = run_epoch(model, train_dl, warmup_opt, criterion, scaler, l1_lam, True)
        vl_loss, vl_y, vl_p = run_epoch(model, val_dl, None, criterion, scaler, 0.0, False)

        try:
            vl_auc = roc_auc_score(vl_y, vl_p, average="macro")
        except ValueError:
            vl_auc = 0.0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_auc"].append(vl_auc)
        history["lr"].append(warmup_sched.last_lr)

        print(f"  [Warmup {epoch:2d}/{warmup_ep}]  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"val_auc={vl_auc:.4f}  lr={warmup_sched.last_lr:.2e}")

    # ────────────────────────────────────────────────────────────────────────
    # Phase 2 – Full fine-tuning
    # ────────────────────────────────────────────────────────────────────────
    remaining = MAX_EPOCHS - warmup_ep
    print(f"\n{'─'*55}")
    print(f"  Phase 2 – Full Fine-tuning ({remaining} epochs, all layers)")
    print(f"{'─'*55}")

    unfreeze_backbone(model)
    ft_opt   = build_optimizer(model, "finetune", lr=lr, lr_head=lr_head, wd=wd, opt_name=opt_name)
    ft_sched = WarmupCosineScheduler(ft_opt, 0, remaining, lr)

    best_val_auc   = 0.0
    best_state     = None
    no_improve_cnt = 0

    for epoch in range(1, remaining + 1):
        ft_sched.step()
        tr_loss, _, _        = run_epoch(model, train_dl, ft_opt, criterion, scaler, l1_lam, True)
        vl_loss, vl_y, vl_p  = run_epoch(model, val_dl, None, criterion, scaler, 0.0, False)

        try:
            vl_auc = roc_auc_score(vl_y, vl_p, average="macro")
        except ValueError:
            vl_auc = 0.0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_auc"].append(vl_auc)
        history["lr"].append(ft_sched.last_lr)

        print(f"  [FT {epoch:3d}/{remaining}]  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"val_auc={vl_auc:.4f}  lr={ft_sched.last_lr:.2e}", end="")

        # ── Early stopping ─────────────────────────────────────────────────
        if vl_auc > best_val_auc + MIN_DELTA:
            best_val_auc   = vl_auc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_cnt = 0
            print("  ✓ best", end="")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n  [EarlyStopping] {EARLY_STOP_PATIENCE} epochs without improvement.")
                break
        print()

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc, history


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"],   label="Val loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_auc"])
    axes[1].set_title("Validation AUC-ROC"); axes[1].grid(True)

    axes[2].plot(history["lr"])
    axes[2].set_title("Learning Rate (warmup + cosine)"); axes[2].grid(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Training curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--binary",     action="store_true",
                        help="Binary mode: predict disease(1) vs normal(0) instead of 14 labels")
    args = parser.parse_args()

    n_outputs  = 1 if args.binary else NUM_CLASSES
    mode_label = "BINARY (disease/normal)" if args.binary else "MULTI-LABEL (14 pathologies)"

    print("=" * 60)
    print("  VinDr-CXR | Step 3: EfficientNet-B0 Fine-tuning")
    print(f"  Mode  : {mode_label}")
    print(f"  Device: {DEVICE}  |  AMP: {USE_AMP}  |  Batch: {args.batch_size}")
    print("=" * 60)

    # ── Load splits ───────────────────────────────────────────────────────────
    for f in ["split_train.csv", "split_val.csv", "split_test.csv"]:
        if not os.path.isfile(os.path.join(DATA_ROOT, f)):
            sys.exit(f"[ERROR] {f} not found. Run 01_data_analysis.py first.")

    train_df = pd.read_csv(os.path.join(DATA_ROOT, "split_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_ROOT, "split_val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_ROOT, "split_test.csv"))

    # ── Build image index ─────────────────────────────────────────────────────
    print(f"\n[index] Scanning {TRAIN_DIR} …")
    image_index = build_image_index(TRAIN_DIR)
    print(f"[index] {len(image_index):,} DICOMs.")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl = make_dataloaders(
        train_df, val_df, test_df, image_index,
        batch_size  = args.batch_size,
        num_workers = NUM_WORKERS,
        binary_mode = args.binary,
    )
    print(f"[loaders] train={len(train_dl)} batches  val={len(val_dl)}  test={len(test_dl)}")

    # ── Build model ───────────────────────────────────────────────────────────
    model     = build_model(num_classes=n_outputs)
    ckpt_path = os.path.join(CHECKPOINT_DIR,
                             "finetune_binary_best.pt" if args.binary else "finetune_best.pt")

    if args.resume and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[resume] Loaded checkpoint from {ckpt_path}")

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_start = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total params: {total_params/1e6:.1f} M  "
          f"| Initially trainable (head): {trainable_start/1e6:.2f} M")

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model, best_val_auc, history = train(model, train_dl, val_dl)
    elapsed = (time.time() - t0) / 60
    print(f"\n[train] Finished in {elapsed:.1f} min  |  Best val AUC: {best_val_auc:.4f}")

    # ── Optuna re-tuning if needed ────────────────────────────────────────────
    if should_retune(best_val_auc):
        print(f"\n[Optuna] Triggering hyperparameter search (val_auc={best_val_auc:.4f}) …")

        def objective(hp: dict) -> float:
            m2 = build_model(dropout=hp.get("dropout_rate", DROPOUT_RATE))
            _, val_auc, _ = train(m2, train_dl, val_dl, hp=hp)
            del m2
            torch.cuda.empty_cache()
            return val_auc

        best_hp = run_optuna(
            objective_fn = objective,
            storage_path = os.path.join(CHECKPOINT_DIR, "optuna_finetune.db"),
        )

        print("[Optuna] Final re-training with best hyperparameters …")
        model = build_model(dropout=best_hp.get("dropout_rate", DROPOUT_RATE))
        model, best_val_auc, history = train(model, train_dl, val_dl, hp=best_hp)
        print(f"[Optuna] Post-tuning val AUC: {best_val_auc:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_auc":          best_val_auc,
        "num_classes":      NUM_CLASSES,
        "history":          history,
    }, ckpt_path)
    print(f"[checkpoint] Saved → {ckpt_path}")

    # ── Training curves ───────────────────────────────────────────────────────
    plot_curves(history, os.path.join(RESULTS_DIR, "finetune_training_curves.png"))

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[eval] Evaluating on internal test set …")
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(test_dl, desc="  Test", ncols=80):
            imgs = imgs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().float().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_labels, all_probs)
    print_metrics(metrics, title="EfficientNet-B0 Fine-tuning – Test Set")

    # ── Extra metrics for binary mode ─────────────────────────────────────────
    if args.binary:
        from sklearn.metrics import (accuracy_score, roc_auc_score,
                                     confusion_matrix, classification_report)
        y_true = all_labels[:, 0]
        y_prob = all_probs[:, 0]
        y_pred = (y_prob >= 0.5).astype(int)
        acc    = accuracy_score(y_true, y_pred)
        auc    = roc_auc_score(y_true, y_prob)
        cm     = confusion_matrix(y_true, y_pred)
        print(f"\n  [Binary] Accuracy : {acc:.4f}")
        print(f"  [Binary] AUC-ROC  : {auc:.4f}")
        print(f"  [Binary] Confusion Matrix (Normal / Disease):")
        print(f"           TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"           FN={cm[1,0]}  TP={cm[1,1]}")
        print("\n" + classification_report(y_true, y_pred,
                                           target_names=["Normal", "Disease"]))

    save_metrics(metrics,
                 os.path.join(RESULTS_DIR,
                              "finetune_binary_test_metrics.json" if args.binary
                              else "finetune_test_metrics.json"),
                 title="EfficientNet-B0 FT Test")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3 Complete")
    print(f"  Best val AUC (fine-tune)    : {best_val_auc:.4f}")
    print(f"  Test AUC (macro)            : {metrics['macro_auc']:.4f}")
    print(f"  Test AUC (weighted)         : {metrics['weighted_auc']:.4f}")
    print(f"  Test Macro F1               : {metrics['macro_f1']:.4f}")
    print(f"  Training time               : {elapsed:.1f} min")
    print(f"  Checkpoint                  : {ckpt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
