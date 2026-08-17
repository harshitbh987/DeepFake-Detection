
from __future__ import annotations
import logging, sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2, numpy as np
sys.path.insert(0, "/home/claude/DeepSentinel")
from config import FACE_CROP_SIZE, FACE_MARGIN_RATIO, MIN_FACE_SIZE, FACE_CONF_THRESH

logger = logging.getLogger(__name__)

@dataclass
class FaceCropResult:
    crop_rgb:        object   # np.ndarray (H,W,3) float32
    bbox:            tuple    # (x1,y1,x2,y2)
    detection_conf:  float
    frame_index:     int


class FaceCropper:
    def __init__(self, target_size=FACE_CROP_SIZE, margin_ratio=FACE_MARGIN_RATIO,
                 conf_threshold=FACE_CONF_THRESH, device="cpu"):
        self.target_size    = target_size
        self.margin_ratio   = margin_ratio
        self.conf_threshold = conf_threshold
        self.device         = device
        self._detector      = self._init_detector()

    def _init_detector(self):
        try:
            from facenet_pytorch import MTCNN
            det = MTCNN(keep_all=False, device=self.device,
                        min_face_size=MIN_FACE_SIZE, post_process=False, select_largest=True)
            logger.info("MTCNN ready on %s", self.device)
            return det
        except Exception as e:
            logger.warning("MTCNN failed (%s), using Haar fallback", e)
            xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            return cv2.CascadeClassifier(xml)

    def crop_largest_face(self, bgr_frame, frame_index=0):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = bgr_frame.shape[:2]
        try:
            if hasattr(self._detector, "detect"):
                return self._mtcnn_detect(rgb, h, w, frame_index)
            else:
                return self._haar_detect(bgr_frame, rgb, h, w, frame_index)
        except Exception as e:
            logger.debug("Face detection error frame %d: %s", frame_index, e)
            return None

    def _mtcnn_detect(self, rgb, h, w, frame_index):
        from PIL import Image as PI
        boxes, probs = self._detector.detect(PI.fromarray(rgb))
        if boxes is None or len(boxes) == 0: return None
        conf = float(probs[0]) if probs is not None else 0.0
        if conf < self.conf_threshold: return None
        x1,y1,x2,y2 = [int(v) for v in boxes[0]]
        crop = self._margin_crop(rgb, x1, y1, x2, y2, h, w)
        if crop is None: return None
        return FaceCropResult(crop_rgb=crop, bbox=(x1,y1,x2,y2), detection_conf=conf, frame_index=frame_index)

    def _haar_detect(self, bgr, rgb, h, w, frame_index):
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE,MIN_FACE_SIZE))
        if len(faces) == 0: return None
        x,y,fw,fh = max(faces, key=lambda r: r[2]*r[3])
        crop = self._margin_crop(rgb, x, y, x+fw, y+fh, h, w)
        if crop is None: return None
        return FaceCropResult(crop_rgb=crop, bbox=(x,y,x+fw,y+fh), detection_conf=1.0, frame_index=frame_index)

    def _margin_crop(self, rgb, x1, y1, x2, y2, fh, fw):
        bw, bh = x2-x1, y2-y1
        if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE: return None
        pw, ph = int(bw*self.margin_ratio), int(bh*self.margin_ratio)
        nx1,ny1 = max(0,x1-pw), max(0,y1-ph)
        nx2,ny2 = min(fw,x2+pw), min(fh,y2+ph)
        crop = rgb[ny1:ny2, nx1:nx2]
        if crop.size == 0: return None
        return cv2.resize(crop, (self.target_size,self.target_size),
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    def crop_batch(self, bgr_frames, frame_indices):
        return [self.crop_largest_face(f, i) for f, i in zip(bgr_frames, frame_indices)]
