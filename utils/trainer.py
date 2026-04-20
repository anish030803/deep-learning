import torch, torch.nn as nn, torchvision.models as M, math, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import NUM_CLASSES, DROPOUT_RATE, WEIGHT_DECAY, USE_AMP, GRAD_CLIP_NORM, LR, LR_HEAD
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes=NUM_CLASSES, dropout=DROPOUT_RATE, pretrained=True):
    weights = M.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = M.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(p=dropout*0.5), nn.Linear(512, num_classes))
    return model.to(DEVICE)

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith("classifier"): param.requires_grad_(False)

def unfreeze_backbone(model):
    for param in model.parameters(): param.requires_grad_(True)

def l1_penalty(model, lambda_l1):
    penalty = torch.tensor(0.0, device=DEVICE)
    for name, module in model.named_modules():
        if "classifier" in name and isinstance(module, nn.Linear):
            penalty = penalty + module.weight.abs().sum()
    return lambda_l1 * penalty

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr):
        self.optimizer = optimizer; self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs; self.base_lr = base_lr; self._epoch = 0
    def step(self):
        self._epoch += 1; e = self._epoch
        if e <= self.warmup_epochs: factor = e / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups: pg["lr"] = pg.get("_base_lr", self.base_lr) * factor
        return self.optimizer.param_groups[0]["lr"]
    @property
    def last_lr(self): return self.optimizer.param_groups[0]["lr"]

def build_optimizer(model, phase, lr=LR, lr_head=LR_HEAD, wd=WEIGHT_DECAY, opt_name="adamw"):
    if phase == "warmup":
        params = [{"params": model.classifier.parameters(), "lr": lr_head, "_base_lr": lr_head}]
    else:
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.requires_grad]
        params = [{"params": backbone_params, "lr": lr, "_base_lr": lr}, {"params": list(model.classifier.parameters()), "lr": lr_head, "_base_lr": lr_head}]
    return torch.optim.AdamW(params, weight_decay=wd)

def run_epoch(model, loader, optimizer, criterion, scaler, lambda_l1, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0; all_labels = []; all_probs = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE, non_blocking=True); labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type=="cuda")):
                logits = model(imgs); loss = criterion(logits, labels)
                if is_train: loss = loss + l1_penalty(model, lambda_l1)
            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM); optimizer.step()
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).detach().cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())
    return total_loss/max(len(loader),1), np.concatenate(all_labels), np.concatenate(all_probs)