#!/usr/bin/env python3
"""
train.py — DeepSentinel Model Training
========================================
Train EfficientNet-B0 for deepfake detection in two phases.

Usage:
    python train.py
    python train.py --warmup_epochs 5 --finetune_epochs 20
    python train.py --device cpu
"""
import argparse, os, sys, time, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

from config import (PRIMARY_WEIGHTS, WARMUP_EPOCHS, FINETUNE_EPOCHS,
                    WARMUP_LR, FINETUNE_LR, WEIGHT_DECAY, GRAD_CLIP,
                    LABEL_SMOOTHING, DATASET_ROOT)
from models.model_loader import SentinelNet, persist_checkpoint
from training.dataset_builder import make_dataloaders

# PyTorch 2.6 safe-load fix
try:
    import numpy._core.multiarray as _nca
    torch.serialization.add_safe_globals([_nca.scalar])
except Exception:
    pass
_orig_load = torch.load
def _safe_load(f, map_location=None, **kw):
    kw['weights_only'] = False
    return _orig_load(f, map_location=map_location, **kw)
torch.load = _safe_load


class SmoothedBCE(nn.Module):
    def __init__(self, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.s = smoothing
        self._bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        return self._bce(logits, targets * (1 - self.s) + self.s / 2)


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.float().view(-1, 1).to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
            total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return avg_loss, auc


def main():
    parser = argparse.ArgumentParser(description='Train DeepSentinel')
    parser.add_argument('--warmup_epochs',   type=int,   default=WARMUP_EPOCHS)
    parser.add_argument('--finetune_epochs', type=int,   default=FINETUNE_EPOCHS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', default=DATASET_ROOT)
    args = parser.parse_args()

    print(f"Device  : {args.device}")
    print(f"Dataset : {args.dataset}")

    train_loader, val_loader = make_dataloaders(args.dataset)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model     = SentinelNet(pretrained=True).to(args.device)
    criterion = SmoothedBCE()
    log_rows  = []
    best_auc  = 0.0

    # ── Phase 1: Warm-up ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("Phase 1: Warm-up (backbone frozen)")
    print("="*55)
    model.freeze_backbone()
    opt_wu = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=WARMUP_LR, weight_decay=WEIGHT_DECAY)
    sch_wu = CosineAnnealingLR(opt_wu, T_max=args.warmup_epochs, eta_min=WARMUP_LR/10)

    for epoch in range(1, args.warmup_epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion, opt_wu, args.device, True)
        va_loss, va_auc = run_epoch(model, val_loader,   criterion, opt_wu, args.device, False)
        sch_wu.step()
        print(f"  Epoch {epoch:02d}/{args.warmup_epochs} | "
              f"tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f} | "
              f"va_loss={va_loss:.4f} va_auc={va_auc:.4f} | {time.time()-t0:.1f}s")
        log_rows.append(dict(phase='warmup', epoch=epoch,
                             tr_loss=round(tr_loss,6), tr_auc=round(tr_auc,6),
                             va_loss=round(va_loss,6), va_auc=round(va_auc,6)))
        if va_auc > best_auc:
            best_auc = va_auc
            persist_checkpoint(model, PRIMARY_WEIGHTS, {'best_auc': best_auc})
            print(f"  ✅ New best AUC {best_auc:.4f} — saved")

    # ── Phase 2: Fine-tuning ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("Phase 2: Fine-tuning (top blocks unfrozen)")
    print("="*55)
    model.unfreeze_top_blocks(num_blocks=3)
    opt_ft = AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    sch_ft = CosineAnnealingLR(opt_ft, T_max=args.finetune_epochs, eta_min=FINETUNE_LR/10)

    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion, opt_ft, args.device, True)
        va_loss, va_auc = run_epoch(model, val_loader,   criterion, opt_ft, args.device, False)
        sch_ft.step()
        print(f"  Epoch {epoch:02d}/{args.finetune_epochs} | "
              f"tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f} | "
              f"va_loss={va_loss:.4f} va_auc={va_auc:.4f} | {time.time()-t0:.1f}s")
        log_rows.append(dict(phase='finetune', epoch=epoch,
                             tr_loss=round(tr_loss,6), tr_auc=round(tr_auc,6),
                             va_loss=round(va_loss,6), va_auc=round(va_auc,6)))
        if va_auc > best_auc:
            best_auc = va_auc
            persist_checkpoint(model, PRIMARY_WEIGHTS, {'best_auc': best_auc})
            print(f"  ✅ New best AUC {best_auc:.4f} — saved")

    # ── Save log ──────────────────────────────────────────────────────────
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_log.csv')
    with open(log_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)
    print(f"\nBest AUC       : {best_auc:.4f}")
    print(f"Checkpoint     : {PRIMARY_WEIGHTS}")
    print(f"Training log   : {log_path}")
    print("\nRun:  python plot_curves.py  to visualize results.")

if __name__ == '__main__':
    main()
