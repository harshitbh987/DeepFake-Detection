
import sys
sys.path.insert(0, "/home/claude/DeepSentinel")

import io, json, logging, os, time
from pathlib import Path
import cv2, numpy as np
import streamlit as st

from config import PRIMARY_WEIGHTS, FRAMES_PER_VIDEO, FAKE_DECISION_BOUNDARY, ALLOWED_VIDEO_EXT, TEMP_UPLOAD_DIR
from detector.inference_engine import InferenceEngine
from visualization.confidence_plot import frame_score_chart, score_gauge, score_timeline, build_rgb_thumbnail

logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="DeepSentinel", page_icon="🔍", layout="wide")
st.markdown("""
<style>
.verdict-box{border-radius:14px;padding:1.4rem 2rem;text-align:center;margin:1rem 0;border:2px solid;}
.verdict-fake{background:#450a0a;border-color:#ef4444;}
.verdict-real{background:#052e16;border-color:#22c55e;}
.verdict-title{font-size:2.4rem;font-weight:800;}
.verdict-sub{font-size:.95rem;color:#94a3b8;margin-top:.3rem;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔍 DeepSentinel")
    st.caption("Deepfake Detection · EfficientNet-B0")
    st.divider()
    model_path     = st.text_input("Model checkpoint", value=PRIMARY_WEIGHTS)
    num_frames     = st.slider("Frames to sample", 4, 24, FRAMES_PER_VIDEO, 2)
    decision_thresh = st.slider("Decision threshold", 0.30, 0.80, FAKE_DECISION_BOUNDARY, 0.05)
    enable_gradcam  = st.checkbox("Enable Grad-CAM 🔥", value=False)
    st.divider()
    st.markdown("🟢 Real  |  🔴 Fake  |  ⚪ No face")

@st.cache_resource(show_spinner="Loading SentinelNet…")
def get_engine(mp, nf, gc):
    try:
        return InferenceEngine(model_path=mp, enable_gradcam=gc, num_frames=nf)
    except FileNotFoundError as e:
        st.error(str(e)); return None

st.title("🔍 DeepSentinel — Deepfake Detection")
st.markdown("Upload a video. DeepSentinel samples **12 frames**, detects faces, and gives a **REAL / FAKE** verdict.")
st.divider()

uploaded = st.file_uploader("Upload video (MP4 / AVI / MOV / MKV)", type=ALLOWED_VIDEO_EXT)
analyse_btn = st.button("▶️  Analyse Video", type="primary", disabled=uploaded is None, use_container_width=True)

if uploaded and analyse_btn:
    suffix   = Path(uploaded.name).suffix
    tmp_path = os.path.join(TEMP_UPLOAD_DIR, f"upload_{int(time.time())}{suffix}")
    with open(tmp_path, "wb") as f: f.write(uploaded.getbuffer())

    engine = get_engine(model_path, num_frames, enable_gradcam)
    if engine is None: st.stop()

    prog = st.progress(0, "Sampling frames…")
    try:
        prog.progress(30, "Detecting faces…")
        t0 = time.perf_counter()
        verdict, analyses = engine.analyse_video(tmp_path, enable_gradcam=enable_gradcam)
        elapsed = time.perf_counter() - t0
        prog.progress(100, "Done ✅"); time.sleep(0.3); prog.empty()
    except Exception as e:
        prog.empty(); st.error(f"Failed: {e}"); os.remove(tmp_path); st.stop()
    finally:
        try: os.remove(tmp_path)
        except: pass

    st.divider(); st.markdown("## 📊 Results")
    is_fake  = verdict.weighted_score >= decision_thresh
    box_cls  = "verdict-fake" if is_fake else "verdict-real"
    v_colour = "#ef4444" if is_fake else "#22c55e"
    icon     = "🔴" if is_fake else "🟢"
    st.markdown(
        f'<div class="verdict-box {box_cls}">'
        f'<div class="verdict-title" style="color:{v_colour}">{icon} {verdict.verdict}</div>'
        f'<div class="verdict-sub">Score: <b>{verdict.weighted_score:.3f}</b> · '
        f'Confidence: <b>{verdict.confidence_tier}</b> · '
        f'Faces: <b>{verdict.frames_with_face}/{verdict.frames_analysed}</b> · '
        f'Time: <b>{elapsed:.1f}s</b></div></div>',
        unsafe_allow_html=True
    )

    col_g, col_b = st.columns([1, 1.6])
    with col_g:
        st.markdown("#### Score Meter")
        st.plotly_chart(score_gauge(verdict), use_container_width=True)
    with col_b:
        st.markdown("#### Per-Frame Probabilities")
        st.plotly_chart(frame_score_chart(verdict), use_container_width=True)

    st.markdown("#### Score Timeline")
    st.plotly_chart(score_timeline(verdict), use_container_width=True)

    st.markdown("#### Sampled Frames")
    cols = st.columns(4)
    score_map = {s.frame_index: s.fake_prob for s in verdict.frame_scores}
    for i, analysis in enumerate(analyses):
        prob = score_map.get(analysis.frame_index)
        display_bgr = (analysis.heatmap_bgr if (enable_gradcam and analysis.heatmap_bgr is not None)
                       else (cv2.cvtColor(analysis.crop_result.crop_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                             if analysis.crop_result else analysis.bgr_frame))
        thumb = build_rgb_thumbnail(display_bgr, prob, analysis.frame_index)
        with cols[i % 4]:
            st.image(thumb, use_container_width=True)
            icon2 = "🔴" if (prob and prob >= decision_thresh) else ("🟢" if prob else "⚪")
            st.caption(f"{icon2} Frame {analysis.frame_index}" + (f" · {prob:.1%}" if prob else " · No face"))

    with st.expander("📋 Frame Score Details"):
        st.dataframe(engine.score_mgr.export_scores(), use_container_width=True, hide_index=True)
        json_out = json.dumps({"verdict": verdict.verdict, "score": verdict.weighted_score,
                               "frames": engine.score_mgr.export_scores()}, indent=2)
        st.download_button("⬇️ Export JSON", json_out,
                           file_name=f"deepsentinel_{Path(uploaded.name).stem}.json", mime="application/json")
elif not uploaded:
    st.info("👆 Upload a video file to begin detection.", icon="📹")
