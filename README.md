# EfficientNet-B0 – VinDr-CXR (PNG version)

**Model:** EfficientNet-B0 (ImageNet pretrained → fine-tuned)  
**Dataset:** VinDr-CXR PNG images (~2 GB) from Google Drive  
**Task:** Multi-label classification of 14 chest X-ray pathologies  
**Split:** 80% train / 10% val / 10% test  |  Seed: 42  
**Hardware:** NVIDIA RTX 5070 Ti + 64 GB RAM

---

## Setup (one-time)

### 1. Clone the repo
```bash
git clone https://github.com/anish030803/deep-learning.git
cd deep-learning
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install PyTorch (RTX 5070 Ti requires nightly)
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Copy train.csv into the data folder
```
deep-learning/
└── data/
    └── train.csv    ← place it here manually
```

---

## Running the pipeline

### Step 1 — Download PNGs + EDA + Clean + Split
```bash
python 01_data_analysis.py
# Skip download if train_png/ already exists:
python 01_data_analysis.py --no-download
```

### Step 2 — Feature extraction (frozen backbone)
```bash
python 02_feature_extraction.py
```

### Step 3 — Full fine-tuning
```bash
# Multi-label (14 pathologies):
python 03_train_finetune.py

# Binary (disease vs normal):
python 03_train_finetune.py --binary
```

### Step 4 — 5-fold cross-validation
```bash
python 04_cross_validation.py
```

---

## Key settings (configs/config.py)

| Parameter | Value |
|-----------|-------|
| SEED | 42 |
| IMG_SIZE | 224 |
| BATCH_SIZE | 32 |
| MAX_EPOCHS | 20 |
| WARMUP_EPOCHS | 5 |
| TRAIN/VAL/TEST | 80/10/10 |
| LR (backbone) | 1e-4 |
| LR (head) | 1e-3 |
| Early stop patience | 7 |

---

## Outputs
```
outputs/
├── checkpoints/
│   ├── finetune_best.pt
│   └── finetune_binary_best.pt
└── results/
    ├── class_distribution.png
    ├── cooccurrence_matrix.png
    ├── finetune_training_curves.png
    ├── finetune_test_metrics.json
    └── cv_summary.json
```
