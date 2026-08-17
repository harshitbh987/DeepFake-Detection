
from __future__ import annotations
import glob, logging, os, random, sys
from typing import List, Tuple
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
sys.path.insert(0, "/home/claude/DeepSentinel")
from config import FRAMES_PER_VIDEO_TRAIN, BATCH_SIZE, NUM_WORKERS, VAL_SPLIT
from detector.frame_sampler import FrameSampler
from detector.face_cropper  import FaceCropper
from models.model_loader    import build_training_transform, build_inference_transform, FACE_CROP_SIZE

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
logger = logging.getLogger(__name__)


class DeepfakeVideoDataset(Dataset):
    def __init__(self, video_entries, transform, use_jitter=True, max_retries=3):
        self.entries   = video_entries
        self.transform = transform
        self.sampler   = FrameSampler(num_frames=FRAMES_PER_VIDEO_TRAIN, use_jitter=use_jitter, jitter_range=4)
        self.cropper   = FaceCropper()
        self.retries   = max_retries

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        return self._load_crop(path), torch.tensor([label], dtype=torch.float32)

    def _load_crop(self, video_path):
        from config import FACE_CROP_SIZE
        for _ in range(self.retries):
            try:
                indices, frames = self.sampler.sample(video_path)
                if not frames: continue
                pick  = random.randint(0, len(frames)-1)
                crop  = self.cropper.crop_largest_face(frames[pick], indices[pick])
                if crop is None: continue
                return self.transform(crop.crop_rgb.astype(np.uint8))
            except Exception: pass
        return torch.zeros(3, FACE_CROP_SIZE, FACE_CROP_SIZE)


def discover_videos(dataset_root):
    entries = []
    for cls, label in [("real", 0.0), ("fake", 1.0)]:
        d = os.path.join(dataset_root, cls)
        if not os.path.isdir(d): continue
        for ext in VIDEO_EXT:
            for p in glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True):
                entries.append((p, label))
    logger.info("Found %d videos in %s", len(entries), dataset_root)
    return entries


def split_entries(entries, val_ratio=VAL_SPLIT, seed=42):
    rng  = random.Random(seed)
    real = [e for e in entries if e[1]==0.0]; rng.shuffle(real)
    fake = [e for e in entries if e[1]==1.0]; rng.shuffle(fake)
    def _s(lst): cut=max(1,int(len(lst)*val_ratio)); return lst[cut:], lst[:cut]
    rt,rv = _s(real); ft,fv = _s(fake)
    return rt+ft, rv+fv


def build_weighted_sampler(entries):
    labels = [e[1] for e in entries]
    nr, nf = labels.count(0.0), labels.count(1.0)
    total  = len(labels)
    wr = total/(2*nr) if nr>0 else 1.0
    wf = total/(2*nf) if nf>0 else 1.0
    weights = [wr if l==0.0 else wf for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def make_dataloaders(dataset_root):
    all_e = discover_videos(dataset_root)
    tr_e, va_e = split_entries(all_e)
    tr_ds = DeepfakeVideoDataset(tr_e, build_training_transform(), use_jitter=True)
    va_ds = DeepfakeVideoDataset(va_e, build_inference_transform(), use_jitter=False)
    sampler = build_weighted_sampler(tr_e)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    logger.info("DataLoaders: train=%d, val=%d batches", len(tr_dl), len(va_dl))
    return tr_dl, va_dl
