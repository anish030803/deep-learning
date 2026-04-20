"""
04_cross_validation.py
======================
Step 4: 5-fold stratified cross-validation for EfficientNet-B0.

For each fold k (1…5):
  1. Build train_k / val_k DataLoaders from the cleaned label CSV
     using StratifiedKFold on the primary pathology label.
  2. Train a fresh EfficientNet-B0 with the same warmup + fine-tuning
     pipeline as 03_train_finetune.py.
  3. Evaluate on val_k → collect per-class AUC, F1, AP.
  4. Save fold checkpoint to outputs/checkpoints/fold_{k}_best.pt

After all folds:
  5. Aggregate per-fold metrics → mean ± std for every metric.
  6. Save:
       outputs/results/cv_per_fold_metrics.json
       outputs/results/cv_summary.json
       outputs/results/cv_auc_boxplot.png

Usage:
    python 04_cross_validation.py [--folds 5] [--fast]

Flags:
    --folds N   Number of folds (default: 5)
    --fast      Use fewer epochs per fold (for quick sanity-checks)
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from configs.config import (
    SEED, NUM_CLASSES, DATA_ROOT, TRAIN_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    CV_N_FOLDS, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_AMP, GRAD_CLIP_NORM,
    LR, LR_HEAD, WEIGHT_DECAY, L1_LAMBDA, DROPOUT_RATE,
    MAX_EPOCHS, WARMUP_EPOCHS, EARLY_STOP_PATIENCE, MIN_DELTA,
)
from utils.image_utils import build_image_index
from utils.dataset import VinDrDataset, get_transforms, make_dataloaders
from utils.metrics import compute_metrics, print_metrics, save_metrics, PATHOLOGY_NAMES
from utils.trainer import (
    build_model, freeze_backbone, unfreeze_backbone,
    WarmupCosineScheduler, build_optimizer, run_epoch, l1_penalty,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold_idx:    int,
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    image_index: dict,
    max_epochs:  int = MAX_EPOCHS,
) -> tuple[float, dict]:
    """
    Train EfficientNet-B0 on one fold.

    Returns (best_val_auc, per_class_metrics_dict).
    """
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    label_cols   = [f"label_{c}" for c in range(14)]
    label_matrix = train_df[label_cols].values.astype(np.float32)

    # ── Weighted sampler ──────────────────────────────────────────────────────
    class_freq = label_matrix.mean(axis=0)
    def _sw(row):
        pos = np.where(row > 0)[0]
        return 1.0 / (class_freq[pos].mean() + 1e-6) if len(pos) > 0 else 1.0 / (class_freq.max() + 1e-6)
    weights = torch.DoubleTensor([_sw(label_matrix[i]) for i in range(len(train_df))])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ds = VinDrDataset(train_df, image_index, split="train")
    val_ds   = VinDrDataset(val_df,   image_index, split="val")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=(NUM_WORKERS > 0))
    val_dl   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=(NUM_WORKERS > 0))

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model()
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.type == "cuda") else None

    best_val_auc   = 0.0
    best_state     = None
    no_improve_cnt = 0

    # ── Phase 1: Warmup ───────────────────────────────────────────────────────
    freeze_backbone(model)
    w_opt   = build_optimizer(model, "warmup", lr_head=LR_HEAD, wd=WEIGHT_DECAY)
    w_sched = WarmupCosineScheduler(w_opt, WARMUP_EPOCHS, WARMUP_EPOCHS, LR_HEAD)

    print(f"\n  [Fold {fold_idx}] Warmup ({WARMUP_EPOCHS} epochs) …")
    for ep in range(1, WARMUP_EPOCHS + 1):
        w_sched.step()
        run_epoch(model, train_dl, w_opt, criterion, scaler, L1_LAMBDA, True)

    # ── Phase 2: Full fine-tune ───────────────────────────────────────────────
    remaining = max_epochs - WARMUP_EPOCHS
    unfreeze_backbone(model)
    ft_opt   = build_optimizer(model, "finetune", lr=LR, lr_head=LR_HEAD, wd=WEIGHT_DECAY)
    ft_sched = WarmupCosineScheduler(ft_opt, 0, remaining, LR)

    print(f"  [Fold {fold_idx}] Fine-tuning ({remaining} epochs) …")
    for ep in range(1, remaining + 1):
        ft_sched.step()
        tr_loss, _, _       = run_epoch(model, train_dl, ft_opt, criterion, scaler, L1_LAMBDA, True)
        vl_loss, vl_y, vl_p = run_epoch(model, val_dl,   None,  criterion, scaler, 0.0, False)

        try:
            vl_auc = roc_auc_score(vl_y, vl_p, average="macro")
        except ValueError:
            vl_auc = 0.0

        print(f"    ep={ep:3d}  tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  vl_auc={vl_auc:.4f}",
              end="")

        if vl_auc > best_val_auc + MIN_DELTA:
            best_val_auc   = vl_auc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_cnt = 0
            print("  ✓", end="")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n  [EarlyStopping] Fold {fold_idx} stopped at epoch {ep}.")
                break
        print()

    # ── Restore best & evaluate ───────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    # Save fold checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold_{fold_idx}_best.pt")
    torch.save({"model_state_dict": model.state_dict(), "val_auc": best_val_auc}, ckpt_path)
    print(f"  [Fold {fold_idx}] checkpoint → {ckpt_path}")

    # Full metrics on val_k
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in val_dl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    metrics = compute_metrics(all_labels, all_probs)
    print_metrics(metrics, title=f"Fold {fold_idx} Validation")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return best_val_auc, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate cross-validation results
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_cv_results(fold_metrics: list[dict]) -> dict:
    """
    Compute mean ± std across folds for all scalar metrics.
    """
    def _nanmean(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")
    def _nanstd(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.std(vals))  if vals else float("nan")

    summary: dict = {}

    # Global scalar metrics
    for key in ["macro_auc", "weighted_auc", "macro_f1", "macro_precision", "macro_recall"]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": _nanmean(vals), "std": _nanstd(vals), "folds": vals}

    # Per-class AUC
    all_class_names = list(fold_metrics[0]["per_class_auc"].keys())
    per_class_auc_summary: dict = {}
    for name in all_class_names:
        vals = [m["per_class_auc"].get(name, float("nan")) for m in fold_metrics]
        per_class_auc_summary[name] = {"mean": _nanmean(vals), "std": _nanstd(vals)}
    summary["per_class_auc"] = per_class_auc_summary

    # Per-class F1
    per_class_f1_summary: dict = {}
    for name in all_class_names:
        vals = [m["per_class_f1"].get(name, float("nan")) for m in fold_metrics]
        per_class_f1_summary[name] = {"mean": _nanmean(vals), "std": _nanstd(vals)}
    summary["per_class_f1"] = per_class_f1_summary

    return summary


def plot_cv_results(fold_metrics: list[dict], n_folds: int, save_path: str) -> None:
    """AUC box plot + bar chart across folds."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Box plot: per-fold macro AUC ──────────────────────────────────────────
    macro_aucs = [m["macro_auc"] for m in fold_metrics]
    axes[0].bar([f"Fold {i+1}" for i in range(n_folds)], macro_aucs, color="#3498db", edgecolor="white")
    axes[0].axhline(np.mean(macro_aucs), color="red", linestyle="--", label=f"Mean={np.mean(macro_aucs):.4f}")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Macro AUC per Fold (EfficientNet-B0)\n{n_folds}-fold CV")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.5)
    for i, v in enumerate(macro_aucs):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    # ── Per-class AUC mean ± std ──────────────────────────────────────────────
    all_class_names = list(fold_metrics[0]["per_class_auc"].keys())
    means = []
    stds  = []
    for name in all_class_names:
        vals  = [m["per_class_auc"].get(name, float("nan")) for m in fold_metrics]
        valid = [v for v in vals if not np.isnan(v)]
        means.append(float(np.mean(valid)) if valid else 0.0)
        stds.append(float(np.std(valid))   if valid else 0.0)

    y_pos = range(len(all_class_names))
    axes[1].barh(list(y_pos), means, xerr=stds, align="center",
                 color="#2ecc71", ecolor="#e74c3c", capsize=4)
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(all_class_names, fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Per-class AUC (mean ± std across folds)")
    axes[1].axvline(0.5, color="gray", linestyle=":")
    axes[1].grid(axis="x", alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[cv] Plot saved → {save_path}")


def print_cv_summary(summary: dict, n_folds: int) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  {n_folds}-Fold Cross-Validation Summary – EfficientNet-B0")
    print(sep)
    for key in ["macro_auc", "weighted_auc", "macro_f1", "macro_precision", "macro_recall"]:
        m = summary[key]["mean"]
        s = summary[key]["std"]
        print(f"  {key:<20} {m:.4f} ± {s:.4f}")

    print(f"\n  Per-class AUC (mean ± std):")
    for name, vs in summary["per_class_auc"].items():
        bar = "█" * int(vs["mean"] * 20)
        print(f"    {name:<28} {vs['mean']:.4f} ± {vs['std']:.4f}  {bar}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int,   default=CV_N_FOLDS)
    parser.add_argument("--fast",  action="store_true",
                        help="Reduce MAX_EPOCHS to 10 for quick testing")
    args = parser.parse_args()

    n_folds    = args.folds
    max_epochs = 10 if args.fast else MAX_EPOCHS

    print("=" * 60)
    print(f"  VinDr-CXR | Step 4: {n_folds}-Fold Stratified Cross-Validation")
    print(f"  Max epochs/fold: {max_epochs}  |  Device: {DEVICE}")
    print("=" * 60)

    # ── Load cleaned labels ───────────────────────────────────────────────────
    cleaned_csv = os.path.join(DATA_ROOT, "cleaned_labels.csv")
    if not os.path.isfile(cleaned_csv):
        sys.exit(f"[ERROR] {cleaned_csv} not found. Run 01_data_analysis.py first.")

    full_df = pd.read_csv(cleaned_csv)
    print(f"[cv] Full dataset: {len(full_df):,} images")

    # ── Build image index ─────────────────────────────────────────────────────
    print(f"[index] Scanning {TRAIN_DIR} …")
    image_index = build_image_index(TRAIN_DIR)
    print(f"[index] {len(image_index):,} DICOMs.")

    # ── Stratification key ────────────────────────────────────────────────────
    label_cols = [f"label_{c}" for c in range(14)]
    label_mat  = full_df[label_cols].values

    def strat_key(row):
        pos = np.where(row > 0)[0]
        return int(pos[0]) if len(pos) > 0 else 14

    strat_keys = np.array([strat_key(label_mat[i]) for i in range(len(full_df))])

    # ── K-Fold split ──────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(full_df)), strat_keys))

    # ── Per-fold training ─────────────────────────────────────────────────────
    fold_aucs    = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n{'━'*60}")
        print(f"  FOLD {fold_idx}/{n_folds}")
        print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")
        print(f"{'━'*60}")

        tr_df  = full_df.iloc[train_idx].reset_index(drop=True)
        vl_df  = full_df.iloc[val_idx].reset_index(drop=True)

        best_auc, metrics = train_fold(
            fold_idx    = fold_idx,
            train_df    = tr_df,
            val_df      = vl_df,
            image_index = image_index,
            max_epochs  = max_epochs,
        )
        fold_aucs.append(best_auc)
        fold_metrics.append(metrics)

        print(f"  [Fold {fold_idx}] Best val AUC = {best_auc:.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate_cv_results(fold_metrics)
    print_cv_summary(summary, n_folds)

    # ── Save results ──────────────────────────────────────────────────────────
    def _fix_nan(obj):
        if isinstance(obj, float) and (obj != obj):   # nan check
            return None
        if isinstance(obj, dict):
            return {k: _fix_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fix_nan(v) for v in obj]
        return obj

    per_fold_path = os.path.join(RESULTS_DIR, "cv_per_fold_metrics.json")
    with open(per_fold_path, "w") as f:
        json.dump(_fix_nan([{"fold": i+1, **m} for i, m in enumerate(fold_metrics)]), f, indent=2)
    print(f"[cv] Per-fold metrics → {per_fold_path}")

    summary_path = os.path.join(RESULTS_DIR, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(_fix_nan(summary), f, indent=2)
    print(f"[cv] Summary → {summary_path}")

    plot_cv_results(fold_metrics, n_folds,
                    os.path.join(RESULTS_DIR, "cv_auc_boxplot.png"))

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 4 Complete")
    print(f"  Fold AUCs : {' | '.join(f'{a:.4f}' for a in fold_aucs)}")
    print(f"  Mean AUC  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
