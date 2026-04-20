"""
02_feature_extraction.py
========================
Step 2: Extract deep features using a frozen EfficientNet-B0 backbone,
        then train a lightweight classifier head on the saved embeddings.

Pipeline:
  A. FEATURE EXTRACTION PASS
     • Load EfficientNet-B0 (ImageNet pretrained), remove the classifier head.
     • Freeze ALL backbone weights.
     • Forward-pass every train/val/test image through the backbone.
     • Save embeddings to disk:
           outputs/features/train_features.npy  (N_train, 1280)
           outputs/features/val_features.npy
           outputs/features/test_features.npy
           outputs/features/train_labels.npy
           outputs/features/val_labels.npy
           outputs/features/test_labels.npy

  B. HEAD TRAINING (on saved features — no GPU forward passes needed)
     • Multi-label linear classifier with:
           – L1 regularisation on head weights (Lasso-style sparsity)
           – L2 weight decay (global)
           – Dropout before head
           – Binary cross-entropy loss (with optional label smoothing)
           – Cosine annealing LR
           – Early stopping
     • Evaluate on val; if val_auc < threshold → trigger Optuna re-tuning.

  C. EVALUATION
     • Print / save per-class AUC-ROC, F1, AP.
     • Save best checkpoint to outputs/checkpoints/fe_head_best.pt

Usage:
    python 02_feature_extraction.py
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from configs.config import (
    SEED, NUM_CLASSES, FEATURE_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    DATA_ROOT, TRAIN_DIR,
    FE_BATCH_SIZE, FE_EPOCHS, FE_LR,
    WEIGHT_DECAY, L1_LAMBDA, DROPOUT_RATE,
    EARLY_STOP_PATIENCE, RETUNE_THRESHOLD,
    NUM_WORKERS, PIN_MEMORY, USE_AMP, GRAD_CLIP_NORM,
)
from utils.image_utils import build_image_index
from utils.dataset import VinDrFeatureDataset
from utils.metrics import compute_metrics, print_metrics, save_metrics
from utils.optuna_tuner import run_optuna, should_retune

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# A. Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone() -> nn.Module:
    """
    Return EfficientNet-B0 with classifier head removed,
    all parameters frozen.
    Output feature dimension: 1280.
    """
    import torchvision.models as M
    weights = M.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = M.efficientnet_b0(weights=weights)

    # Remove the classifier (keep features + avgpool)
    model.classifier = nn.Identity()

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    return model.to(DEVICE).eval()


def extract_features(
    backbone: nn.Module,
    df: pd.DataFrame,
    image_index: dict,
    split: str,
    batch_size: int = FE_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all images in df through the frozen backbone.

    Returns:
        features : (N, 1280) float32
        labels   : (N, 14) float32
    """
    feat_path  = os.path.join(FEATURE_DIR, f"{split}_features.npy")
    label_path = os.path.join(FEATURE_DIR, f"{split}_labels.npy")

    if os.path.isfile(feat_path) and os.path.isfile(label_path):
        print(f"[extract] Loading cached {split} features from disk …")
        return np.load(feat_path), np.load(label_path)

    dataset = VinDrFeatureDataset(df, image_index)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    all_feats  = []
    all_labels = []

    print(f"[extract] Extracting {split} features ({len(df):,} images) …")
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(loader, desc=f"  {split}", ncols=80):
            imgs = imgs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                feats = backbone(imgs)          # (B, 1280)
            all_feats.append(feats.cpu().float().numpy())
            all_labels.append(lbls.numpy())

    features = np.concatenate(all_feats,  axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels, axis=0).astype(np.float32)

    np.save(feat_path,  features)
    np.save(label_path, labels)
    print(f"[extract] Saved {split} features → {feat_path}  shape={features.shape}")
    return features, labels


# ─────────────────────────────────────────────────────────────────────────────
# B. Classifier head
# ─────────────────────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """
    Lightweight head trained on top of frozen EfficientNet-B0 features.

    Architecture:
        Dropout → Linear(1280, 256) → BatchNorm → ReLU →
        Dropout → Linear(256, NUM_CLASSES)
    """
    def __init__(self, in_dim: int = 1280, num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def l1_regularization(model: nn.Module, lambda_l1: float) -> torch.Tensor:
    """Compute L1 penalty on all linear layer weights in the head."""
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            l1_loss = l1_loss + module.weight.abs().sum()
    return lambda_l1 * l1_loss


def train_head(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    hp: dict | None = None,
) -> tuple[MLPHead, float]:
    """
    Train the MLP head on pre-extracted features.

    Returns (best_model, best_val_auc).
    """
    # ── Hyperparameters ───────────────────────────────────────────────────────
    lr             = hp["lr"]            if hp else FE_LR
    wd             = hp["weight_decay"]  if hp else WEIGHT_DECAY
    l1_lam         = hp["l1_lambda"]     if hp else L1_LAMBDA
    dropout        = hp["dropout_rate"]  if hp else DROPOUT_RATE
    label_smooth   = hp.get("label_smoothing", 0.0) if hp else 0.0
    patience       = EARLY_STOP_PATIENCE
    max_epochs     = FE_EPOCHS

    # ── Data ─────────────────────────────────────────────────────────────────
    X_tr = torch.tensor(train_feats,  dtype=torch.float32)
    y_tr = torch.tensor(train_labels, dtype=torch.float32)
    X_vl = torch.tensor(val_feats,    dtype=torch.float32).to(DEVICE)
    y_vl = torch.tensor(val_labels,   dtype=torch.float32).to(DEVICE)

    tr_ds = TensorDataset(X_tr, y_tr)
    tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True,
                       num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = MLPHead(dropout=dropout).to(DEVICE)
    optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)
    bce    = nn.BCEWithLogitsLoss(reduction="mean")

    best_auc       = 0.0
    best_state     = None
    no_improve_cnt = 0
    history        = {"train_loss": [], "val_auc": []}

    print(f"\n[head] Training MLP head for {max_epochs} epochs …")
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)

            # Label smoothing
            if label_smooth > 0:
                yb_smooth = yb * (1 - label_smooth) + 0.5 * label_smooth
            else:
                yb_smooth = yb

            loss = bce(logits, yb_smooth) + l1_regularization(model, l1_lam)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()
            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(tr_dl)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_probs  = torch.sigmoid(val_logits).cpu().numpy()

        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(y_vl.cpu().numpy(), val_probs,
                                    average="macro", multi_class="ovr")
        except ValueError:
            val_auc = 0.0

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_auc > best_auc + 1e-4:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= patience:
                print(f"  [EarlyStopping] No improvement for {patience} epochs. Stopping.")
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  VinDr-CXR | Step 2: Feature Extraction + Head Training")
    print("=" * 60)

    # ── Load split CSVs ───────────────────────────────────────────────────────
    for fname in ["split_train.csv", "split_val.csv", "split_test.csv"]:
        p = os.path.join(DATA_ROOT, fname)
        if not os.path.isfile(p):
            sys.exit(f"[ERROR] {p} not found. Run 01_data_analysis.py first.")

    train_df = pd.read_csv(os.path.join(DATA_ROOT, "split_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_ROOT, "split_val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_ROOT, "split_test.csv"))

    # ── Build image index ─────────────────────────────────────────────────────
    print(f"\n[index] Scanning {TRAIN_DIR} …")
    image_index = build_image_index(TRAIN_DIR)
    print(f"[index] {len(image_index):,} DICOMs found.")

    # ── A. Extract features ───────────────────────────────────────────────────
    backbone = build_backbone()
    print(f"[backbone] EfficientNet-B0 loaded (frozen). Output dim: 1280")

    t0 = time.time()
    tr_feats, tr_labels = extract_features(backbone, train_df, image_index, "train")
    vl_feats, vl_labels = extract_features(backbone, val_df,   image_index, "val")
    te_feats, te_labels = extract_features(backbone, test_df,  image_index, "test")
    print(f"[extract] Total extraction time: {(time.time()-t0)/60:.1f} min")

    del backbone
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── B. Train head ─────────────────────────────────────────────────────────
    head, val_auc = train_head(tr_feats, tr_labels, vl_feats, vl_labels)
    print(f"\n[head] Best val AUC (feature extraction): {val_auc:.4f}")

    # ── Optuna re-tuning if needed ────────────────────────────────────────────
    if should_retune(val_auc):
        print(f"\n[Optuna] val_auc={val_auc:.4f} < {RETUNE_THRESHOLD} → triggering auto-tuning …")

        def objective(hp: dict) -> float:
            _, auc = train_head(tr_feats, tr_labels, vl_feats, vl_labels, hp=hp)
            return auc

        best_hp = run_optuna(
            objective_fn = objective,
            storage_path = os.path.join(CHECKPOINT_DIR, "optuna_fe.db"),
        )
        print("[Optuna] Re-training with best hyperparameters …")
        head, val_auc = train_head(tr_feats, tr_labels, vl_feats, vl_labels, hp=best_hp)
        print(f"[Optuna] Post-tuning val AUC: {val_auc:.4f}")

    # ── C. Evaluate on test set ───────────────────────────────────────────────
    head.eval()
    X_te = torch.tensor(te_feats, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        te_probs = torch.sigmoid(head(X_te)).cpu().numpy()

    metrics = compute_metrics(te_labels, te_probs)
    print_metrics(metrics, title="Feature-Extraction Head – Test Set")
    save_metrics(metrics,
                 os.path.join(RESULTS_DIR, "fe_head_test_metrics.json"),
                 title="FE Head Test")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(CHECKPOINT_DIR, "fe_head_best.pt")
    torch.save({
        "model_state_dict": head.state_dict(),
        "val_auc": val_auc,
        "num_classes": NUM_CLASSES,
    }, ckpt_path)
    print(f"\n[checkpoint] Saved → {ckpt_path}")

    print("\n" + "=" * 60)
    print("  Step 2 Complete")
    print(f"  Feature-extraction head test AUC: {metrics['macro_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
