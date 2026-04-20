"""
Central configuration for EfficientNet-B0 VinDr-CXR PNG pipeline.
RTX 5070 Ti (16 GB VRAM) + 64 GB RAM optimized settings.
"""

import os

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Google Drive dataset ─────────────────────────────────────────────────────
# Folder ID from the shared Google Drive link
GDRIVE_FOLDER_ID = "1WZZKMjr4VBpUmCTyZJ62_BemDBpRzwEj"

# Local root where the dataset will be stored
DATA_ROOT    = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DIR    = os.path.join(DATA_ROOT, "train_png")   # PNGs go here
TRAIN_CSV    = os.path.join(DATA_ROOT, "train.csv")   # you already have this

# ─── Output directories ───────────────────────────────────────────────────────
OUTPUT_ROOT    = os.path.join(os.path.dirname(__file__), "..", "outputs")
FEATURE_DIR    = os.path.join(OUTPUT_ROOT, "features")
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
LOG_DIR        = os.path.join(OUTPUT_ROOT, "logs")
RESULTS_DIR    = os.path.join(OUTPUT_ROOT, "results")

for d in [OUTPUT_ROOT, FEATURE_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset / label settings ─────────────────────────────────────────────────
CLASS_NAMES = [
    "Aortic enlargement",   # 0
    "Atelectasis",          # 1
    "Calcification",        # 2
    "Cardiomegaly",         # 3
    "Consolidation",        # 4
    "ILD",                  # 5
    "Infiltration",         # 6
    "Lung Opacity",         # 7
    "Nodule/Mass",          # 8
    "Other lesion",         # 9
    "Pleural effusion",     # 10
    "Pleural thickening",   # 11
    "Pneumothorax",         # 12
    "Pulmonary fibrosis",   # 13
    "No finding",           # 14
]
NUM_CLASSES         = 14
NO_FINDING_CLASS_ID = 14
LABEL_STRATEGY      = "majority"  # 'majority' or 'union'

# ─── Image preprocessing ──────────────────────────────────────────────────────
IMG_SIZE       = 224
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

# ─── Data splits ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ─── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE          = 32
NUM_WORKERS         = 8
PIN_MEMORY          = True

LR                  = 1e-4
LR_HEAD             = 1e-3
WEIGHT_DECAY        = 1e-4
L1_LAMBDA           = 1e-5
DROPOUT_RATE        = 0.3

MAX_EPOCHS          = 20
WARMUP_EPOCHS       = 5
EARLY_STOP_PATIENCE = 7
MIN_DELTA           = 1e-4

GRAD_CLIP_NORM      = 1.0
USE_AMP             = True

# ─── Feature extraction ───────────────────────────────────────────────────────
FE_BATCH_SIZE = 64
FE_EPOCHS     = 30
FE_LR         = 1e-3

# ─── Optuna ───────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS    = 30
OPTUNA_TIMEOUT     = 3600
RETUNE_THRESHOLD   = 0.80

# ─── Cross-validation ─────────────────────────────────────────────────────────
CV_N_FOLDS = 5

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True

# ─── Classification mode ──────────────────────────────────────────────────────
# BINARY_MODE = True  → 1 output (disease / no disease)
#               False → 14 outputs (multi-label)
BINARY_MODE = False
EARLY_STOP_METRIC = "val_auc"
 