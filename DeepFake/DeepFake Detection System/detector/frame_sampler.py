
from __future__ import annotations
import logging, sys
from typing import List, Optional, Tuple
import numpy as np
sys.path.insert(0, "/home/claude/DeepSentinel")
from config import FRAMES_PER_VIDEO
from utils.video_reader import VideoReader, VideoMeta

logger = logging.getLogger(__name__)

def compute_uniform_indices(total_frames: int, num_samples: int) -> List[int]:
    if total_frames <= 0: return []
    num_samples = min(num_samples, total_frames)
    raw = np.linspace(0, total_frames - 1, num=num_samples, endpoint=True)
    return sorted(set(int(round(i)) for i in raw))

def compute_jittered_indices(total_frames, num_samples, jitter_range=3, seed=None):
    rng  = np.random.default_rng(seed)
    base = compute_uniform_indices(total_frames, num_samples)
    out  = []
    for idx in base:
        shift   = int(rng.integers(-jitter_range, jitter_range + 1))
        clamped = max(0, min(total_frames - 1, idx + shift))
        out.append(clamped)
    return sorted(set(out))


class FrameSampler:
    def __init__(self, num_frames=FRAMES_PER_VIDEO, use_jitter=False, jitter_range=3):
        self.num_frames   = num_frames
        self.use_jitter   = use_jitter
        self.jitter_range = jitter_range

    def sample(self, video_path: str):
        with VideoReader(video_path) as vr:
            meta = vr.meta
            if meta.total_frames < 1: return [], []
            if self.use_jitter:
                indices = compute_jittered_indices(meta.total_frames, self.num_frames, self.jitter_range)
            else:
                indices = compute_uniform_indices(meta.total_frames, self.num_frames)
            pairs = vr.read_frames_at(indices)
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def sample_with_meta(self, video_path: str):
        with VideoReader(video_path) as vr:
            meta    = vr.meta
            indices = compute_uniform_indices(meta.total_frames, self.num_frames)
            pairs   = vr.read_frames_at(indices)
        return [p[0] for p in pairs], [p[1] for p in pairs], meta
