#!/usr/bin/env python3
"""
predict.py — Run DeepSentinel on a single video
Usage:
    python predict.py --video path/to/video.mp4
    python predict.py --video video.mp4 --plot --gradcam
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import numpy._core.multiarray as _nca
    torch.serialization.add_safe_globals([_nca.scalar])
except Exception:
    pass
_orig = torch.load
def _safe(f, map_location=None, **kw):
    kw['weights_only'] = False
    return _orig(f, map_location=map_location, **kw)
torch.load = _safe

from config import PRIMARY_WEIGHTS, FRAMES_PER_VIDEO, FAKE_DECISION_BOUNDARY
from detector.inference_engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description='DeepSentinel — Deepfake Detection')
    parser.add_argument('--video',     required=True)
    parser.add_argument('--model',     default=PRIMARY_WEIGHTS)
    parser.add_argument('--frames',    type=int,   default=FRAMES_PER_VIDEO)
    parser.add_argument('--threshold', type=float, default=FAKE_DECISION_BOUNDARY)
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--plot',      action='store_true')
    parser.add_argument('--gradcam',   action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}"); sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("Run train.py first to generate the checkpoint."); sys.exit(1)

    print(f"Loading model on {args.device}...")
    engine = InferenceEngine(model_path=args.model, device=args.device,
                             enable_gradcam=args.gradcam, num_frames=args.frames)
    print(f"Analysing: {args.video}")
    verdict, analyses = engine.analyse_video(args.video, enable_gradcam=args.gradcam)

    print("\n" + "="*50)
    print(f"  VERDICT    : {verdict.verdict}")
    print(f"  Score      : {verdict.weighted_score:.4f}")
    print(f"  Confidence : {verdict.confidence_tier}")
    print(f"  Faces      : {verdict.frames_with_face}/{verdict.frames_analysed} frames")
    print("="*50)

    if verdict.frame_scores:
        print("\nPer-frame scores:")
        for s in sorted(verdict.frame_scores, key=lambda x: x.frame_index):
            flag = "🔴 FAKE" if s.fake_prob >= args.threshold else "🟢 REAL"
            print(f"  Frame {s.frame_index:4d}: {s.fake_prob:.4f}  {flag}")

    if args.plot and verdict.frame_scores:
        scores = [s.fake_prob for s in verdict.frame_scores]
        frames = [s.frame_index for s in verdict.frame_scores]
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['#EF4444' if s >= args.threshold else '#22C55E' for s in scores]
        ax.bar(frames, scores, color=colors)
        ax.axhline(args.threshold, color='orange', linestyle='--', label=f'Threshold ({args.threshold})')
        ax.axhline(verdict.weighted_score, color='purple', linestyle=':', label=f'Avg ({verdict.weighted_score:.3f})')
        ax.set_ylim(0, 1); ax.set_xlabel('Frame'); ax.set_ylabel('Fake Probability')
        ax.set_title(f'DeepSentinel — {verdict.verdict} ({verdict.weighted_score:.1%})')
        ax.legend(); ax.grid(alpha=0.3)
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out  = f'{stem}_scores.png'
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print(f"\nPlot saved → {out}")

if __name__ == '__main__':
    main()
