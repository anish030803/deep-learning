"""
01_data_analysis.py
===================
Step 1: Download PNGs from Google Drive, parse train.csv,
        run EDA, clean, and split into 80/10/10.

Usage:
    python 01_data_analysis.py
    python 01_data_analysis.py --no-download   # skip if train_png/ already exists
    python 01_data_analysis.py --fast-clean    # skip full integrity check
"""

import os, sys, json, argparse, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from configs.config import (
    GDRIVE_FOLDER_ID, DATA_ROOT, TRAIN_DIR, TRAIN_CSV,
    CLASS_NAMES, LABEL_STRATEGY, SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RESULTS_DIR,
)
from utils.image_utils import build_image_index, load_png

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download from Google Drive
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(skip: bool = False) -> None:
    if skip and os.path.isdir(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 100:
        print(f"[download] Skipping — train_png/ already exists at {TRAIN_DIR}")
        return

    try:
        import gdown
    except ImportError:
        sys.exit("[ERROR] gdown not installed. Run: pip install gdown")

    os.makedirs(DATA_ROOT, exist_ok=True)
    print(f"[download] Downloading PNGs from Google Drive → {TRAIN_DIR}")
    print("[download] Folder ID:", GDRIVE_FOLDER_ID)

    gdown.download_folder(
        id        = GDRIVE_FOLDER_ID,
        output    = TRAIN_DIR,
        quiet     = False,
        use_cookies = False,
    )
    print(f"[download] Complete → {TRAIN_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Check train.csv exists
# ─────────────────────────────────────────────────────────────────────────────

def check_csv() -> None:
    if not os.path.isfile(TRAIN_CSV):
        sys.exit(
            f"\n[ERROR] train.csv not found at {TRAIN_CSV}\n"
            f"Please copy train.csv into the data/ folder:\n"
            f"  {DATA_ROOT}\n"
        )
    print(f"[csv] Found train.csv → {TRAIN_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parse labels
# ─────────────────────────────────────────────────────────────────────────────

def parse_labels(csv_path: str, strategy: str = LABEL_STRATEGY) -> pd.DataFrame:
    """
    Aggregate per-bounding-box rows into per-image multi-hot binary labels.
    Strategy: 'majority' (≥2/3 radiologists) or 'union' (any 1).
    Returns DataFrame: image_id, label_0 … label_13
    """
    print(f"[parse] Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"[parse] Raw rows: {len(df):,} | Images: {df['image_id'].nunique():,}")

    PATHOLOGY_IDS = list(range(14))
    pos_df  = df[df["class_id"].isin(PATHOLOGY_IDS)].copy()
    vote_df = (
        pos_df.groupby(["image_id", "class_id"])["rad_id"]
        .nunique().reset_index()
        .rename(columns={"rad_id": "n_annotators"})
    )

    all_ids    = df["image_id"].unique()
    label_data = {iid: np.zeros(14, dtype=np.int8) for iid in all_ids}
    threshold  = 2 if strategy == "majority" else 1

    for _, row in vote_df.iterrows():
        if row["n_annotators"] >= threshold:
            label_data[row["image_id"]][int(row["class_id"])] = 1

    rows = [{"image_id": iid, **{f"label_{c}": int(label_data[iid][c]) for c in range(14)}}
            for iid in all_ids]

    labels_df = pd.DataFrame(rows)
    print(f"[parse] Aggregated → {len(labels_df):,} images ({strategy} strategy)")
    return labels_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. EDA
# ─────────────────────────────────────────────────────────────────────────────

def eda(labels_df: pd.DataFrame, image_index: dict) -> dict:
    label_cols    = [f"label_{c}" for c in range(14)]
    class_labels  = CLASS_NAMES[:14]
    counts        = labels_df[label_cols].sum().values
    normal_count  = (labels_df[label_cols].sum(axis=1) == 0).sum()

    print("\n[EDA] Class distribution:")
    for i, name in enumerate(class_labels):
        pct = 100 * counts[i] / len(labels_df)
        print(f"  {name:<28} {counts[i]:>5}  ({pct:.1f}%)")
    print(f"  {'No finding':<28} {normal_count:>5}  "
          f"({100*normal_count/len(labels_df):.1f}%)")

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(class_labels, counts, color="#3498db", edgecolor="white")
    ax.set_xlabel("Number of images")
    ax.set_title("VinDr-CXR: Per-class annotation count (majority vote)")
    ax.bar_label(bars, labels=[f" {c:,}" for c in counts], fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # Co-occurrence heatmap
    label_matrix = labels_df[label_cols].values.astype(np.float32)
    co = label_matrix.T @ label_matrix
    np.fill_diagonal(co, 0)
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(co.astype(int), annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, ax=ax2)
    ax2.set_title("Label Co-occurrence Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "cooccurrence_matrix.png"), dpi=150)
    plt.close(fig2)
    print(f"[EDA] Plots saved → {RESULTS_DIR}/")

    # Sample integrity check (200 random PNGs)
    print("\n[EDA] Sampling 200 random PNGs for integrity check …")
    sample_ids = random.sample(list(image_index.keys()), min(200, len(image_index)))
    corrupt, ok_sizes = [], []
    for iid in tqdm(sample_ids, desc="Integrity", ncols=80):
        path = image_index.get(iid)
        try:
            img = load_png(path)
            ok_sizes.append(img.size)
        except Exception as e:
            corrupt.append((iid, str(e)))

    print(f"  Corrupt PNGs : {len(corrupt)}")
    if ok_sizes:
        widths  = [s[0] for s in ok_sizes]
        heights = [s[1] for s in ok_sizes]
        print(f"  Image sizes  : W {min(widths)}–{max(widths)} px  "
              f"H {min(heights)}–{max(heights)} px")

    labels_per_image = label_matrix.sum(axis=1)
    print(f"  Labels/image : mean={labels_per_image.mean():.2f}  "
          f"max={int(labels_per_image.max())}  "
          f"zero={int((labels_per_image==0).sum())}")

    return {
        "total_images":          int(len(labels_df)),
        "class_counts":          {class_labels[i]: int(counts[i]) for i in range(14)},
        "no_finding":            int(normal_count),
        "corrupt_sample":        len(corrupt),
        "mean_labels_per_image": float(labels_per_image.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clean
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(labels_df: pd.DataFrame, image_index: dict) -> pd.DataFrame:
    n_before  = len(labels_df)
    known_ids = set(image_index.keys())
    labels_df = labels_df[labels_df["image_id"].isin(known_ids)].copy()
    print(f"[clean] Removed {n_before - len(labels_df):,} rows with missing PNGs.")

    before_dedup = len(labels_df)
    labels_df    = labels_df.drop_duplicates(subset="image_id").reset_index(drop=True)
    print(f"[clean] Removed {before_dedup - len(labels_df):,} duplicate image_ids.")
    print(f"[clean] Final dataset: {len(labels_df):,} images.")
    return labels_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stratified split 80/10/10
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(labels_df: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    label_cols = [f"label_{c}" for c in range(14)]
    label_mat  = labels_df[label_cols].values

    def strat_key(row):
        pos = np.where(row > 0)[0]
        return int(pos[0]) if len(pos) > 0 else 14

    labels_df       = labels_df.copy()
    labels_df["_s"] = [strat_key(label_mat[i]) for i in range(len(labels_df))]

    train_df, temp_df = train_test_split(
        labels_df, test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED, stratify=labels_df["_s"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED, stratify=temp_df["_s"],
    )
    for df_ in [train_df, val_df, test_df]:
        df_.drop(columns=["_s"], inplace=True)

    print(f"\n[split] Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  VinDr-CXR PNG | Step 1: Data Analysis & Cleaning")
    print(f"  Seed: {SEED}  |  Split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")
    print("=" * 60)

    # 1. Download PNGs
    download_dataset(skip=args.no_download)

    # 2. Check train.csv
    check_csv()

    # 3. Build PNG index
    print(f"\n[index] Scanning {TRAIN_DIR} …")
    image_index = build_image_index(TRAIN_DIR)
    print(f"[index] Found {len(image_index):,} PNG files.")

    # 4. Parse labels
    labels_df = parse_labels(TRAIN_CSV)

    # 5. EDA
    eda_report = eda(labels_df, image_index)
    rpt_path   = os.path.join(RESULTS_DIR, "eda_report.json")
    with open(rpt_path, "w") as f:
        json.dump(eda_report, f, indent=2)
    print(f"[EDA] Report → {rpt_path}")

    # 6. Clean
    cleaned_df = clean_dataset(labels_df, image_index)
    cleaned_path = os.path.join(DATA_ROOT, "cleaned_labels.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"[clean] Saved → {cleaned_path}")

    # 7. Split
    train_df, val_df, test_df = stratified_split(cleaned_df)
    for name, df_ in [("split_train", train_df), ("split_val", val_df), ("split_test", test_df)]:
        path = os.path.join(DATA_ROOT, f"{name}.csv")
        df_.to_csv(path, index=False)
        print(f"[split] Saved → {path}")

    print("\n" + "=" * 60)
    print("  Step 1 Complete ✓")
    print(f"  Total images : {len(cleaned_df):,}")
    print(f"  Train        : {len(train_df):,}  (80%)")
    print(f"  Val          : {len(val_df):,}   (10%)")
    print(f"  Test         : {len(test_df):,}   (10%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
