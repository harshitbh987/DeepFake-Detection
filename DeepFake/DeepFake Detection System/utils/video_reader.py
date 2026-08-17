
from __future__ import annotations
import logging, os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
import cv2, numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VideoMeta:
    file_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float

    @property
    def resolution(self):
        return (self.width, self.height)

    def __str__(self):
        return (f"VideoMeta(frames={self.total_frames}, fps={self.fps:.2f}, "
                f"res={self.width}x{self.height}, dur={self.duration_sec:.1f}s)")


class VideoReader:
    def __init__(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Video not found: {file_path}")
        self._path = file_path
        self._cap  = None
        self._meta = None

    def open(self):
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {self._path}")
        total  = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._meta = VideoMeta(self._path, total, fps, width, height, total/fps if fps>0 else 0.0)
        return self

    def close(self):
        if self._cap: self._cap.release(); self._cap = None

    def __enter__(self): return self.open()
    def __exit__(self, *_): self.close()

    @property
    def meta(self):
        if not self._meta: raise RuntimeError("Call open() first.")
        return self._meta

    def read_frame(self, idx: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else None

    def read_frames_at(self, indices: List[int]):
        return [(i, f) for i in indices if (f := self.read_frame(i)) is not None]


def probe_video(file_path: str) -> VideoMeta:
    with VideoReader(file_path) as vr:
        return vr.meta
