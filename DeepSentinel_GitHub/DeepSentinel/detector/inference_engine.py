
from __future__ import annotations
import logging, sys, time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/claude/DeepSentinel")
from config import FRAMES_PER_VIDEO, FACE_CROP_SIZE, PRIMARY_WEIGHTS, GRADCAM_LAYER, GRADCAM_ALPHA
from detector.frame_sampler import FrameSampler
from detector.face_cropper  import FaceCropper, FaceCropResult
from models.model_loader    import SentinelNet, load_from_checkpoint, build_inference_transform
from utils.score_manager    import ScoreManager, FrameScore, DetectionVerdict

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    frame_index:  int
    bgr_frame:    object
    crop_result:  object
    fake_prob:    object   # float or None
    heatmap_bgr:  object = None


class GradCAMExtractor:
    def __init__(self, model, target_layer_name=GRADCAM_LAYER):
        self.model        = model
        self._activations = None
        self._gradients   = None
        self._hooks       = []
        target = dict(model.feature_extractor.named_modules()).get(target_layer_name)
        if target:
            self._hooks.append(target.register_forward_hook(lambda m,i,o: setattr(self,"_activations",o.detach())))
            self._hooks.append(target.register_full_backward_hook(lambda m,i,o: setattr(self,"_gradients",o[0].detach())))

    def generate(self, tensor):
        if self._activations is None: return None
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        logit  = self.model(tensor)
        self.model.zero_grad()
        logit.backward(torch.ones_like(logit))
        if self._gradients is None: return None
        weights = self._gradients.mean(dim=[2,3], keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, (FACE_CROP_SIZE, FACE_CROP_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(self, cam, crop_rgb):
        hm = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
        bgr = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(bgr, 1-GRADCAM_ALPHA, hm, GRADCAM_ALPHA, 0)

    def detach(self):
        for h in self._hooks: h.remove()


class InferenceEngine:
    def __init__(self, model_path=PRIMARY_WEIGHTS, device=None, enable_gradcam=False, num_frames=FRAMES_PER_VIDEO):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _   = load_from_checkpoint(model_path, device=self.device)
        self.model.eval()
        self.sampler    = FrameSampler(num_frames=num_frames)
        self.cropper    = FaceCropper(device=self.device)
        self.transform  = build_inference_transform()
        self.score_mgr  = ScoreManager()
        self.gradcam    = GradCAMExtractor(self.model) if enable_gradcam else None
        logger.info("InferenceEngine ready (device=%s, gradcam=%s)", self.device, enable_gradcam)

    def analyse_video(self, video_path, enable_gradcam=None):
        use_cam = enable_gradcam if enable_gradcam is not None else (self.gradcam is not None)
        indices, bgr_frames = self.sampler.sample(video_path)
        crop_results = self.cropper.crop_batch(bgr_frames, indices)
        self.score_mgr.reset()
        analyses = []
        for idx, bgr, crop in zip(indices, bgr_frames, crop_results):
            a = self._process_frame(idx, bgr, crop, use_cam)
            analyses.append(a)
            prob = a.fake_prob if a.fake_prob is not None else 0.5
            self.score_mgr.add(FrameScore(idx, prob, a.fake_prob is not None))
        return self.score_mgr.compute_verdict(), analyses

    def _process_frame(self, frame_index, bgr, crop, use_gradcam):
        if crop is None:
            return FrameAnalysis(frame_index, bgr, None, None)
        tensor = self.transform(crop.crop_rgb.astype(np.uint8)).unsqueeze(0).to(self.device)
        with torch.set_grad_enabled(use_gradcam):
            prob = float(torch.sigmoid(self.model(tensor)).item())
        heatmap = None
        if use_gradcam and self.gradcam:
            cam = self.gradcam.generate(tensor)
            if cam is not None: heatmap = self.gradcam.overlay(cam, crop.crop_rgb)
        return FrameAnalysis(frame_index, bgr, crop, prob, heatmap)
