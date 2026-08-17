
import sys, os, base64, json, time, tempfile
import cv2, numpy as np, torch
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, 'os.path.dirname(os.path.abspath(__file__))')
import numpy._core.multiarray
torch.serialization.add_safe_globals([
    numpy._core.multiarray.scalar,
    np.dtype, np.int64, np.float64, np.float32
])
from config import PRIMARY_WEIGHTS, FAKE_DECISION_BOUNDARY, TEMP_UPLOAD_DIR
from detector.inference_engine import InferenceEngine

app  = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

ALLOWED  = {'mp4','avi','mov','mkv','webm'}
_engines = {}   # cache one engine per num_frames value

def get_engine(num_frames=12):
    global _engines
    if num_frames not in _engines:
        try:
            _engines[num_frames] = InferenceEngine(
                model_path=PRIMARY_WEIGHTS,
                enable_gradcam=False,
                num_frames=num_frames
            )
        except Exception as e:
            print(f'Engine error: {e}')
            return None
    return _engines[num_frames]

def to_b64(bgr, q=82):
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ''

def make_thumb(analysis, prob, size=200):
    if analysis.heatmap_bgr is not None:
        src = analysis.heatmap_bgr
    elif analysis.crop_result is not None:
        src = cv2.cvtColor(analysis.crop_result.crop_rgb.astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
    else:
        src = analysis.bgr_frame
    th  = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    col = ((40,40,220) if (prob and prob >= FAKE_DECISION_BOUNDARY)
           else ((40,200,40) if prob else (100,100,100)))
    th  = cv2.copyMakeBorder(th,3,3,3,3,cv2.BORDER_CONSTANT,value=col)
    return to_b64(th)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status':'ok','model_loaded': bool(_engines or PRIMARY_WEIGHTS)})

@app.route('/analyse', methods=['POST'])
def analyse():
    num_frames = int(request.form.get('num_frames', 12))
    num_frames = max(4, min(24, num_frames))

    eng = get_engine(num_frames)
    if eng is None:
        return jsonify({'error':'Model not loaded. Run training first.'}), 503
    if 'video' not in request.files:
        return jsonify({'error':'No video file (field: video)'}), 400
    f = request.files['video']
    if not f.filename or f.filename.rsplit('.',1)[-1].lower() not in ALLOWED:
        return jsonify({'error':'Unsupported file type'}), 415

    suffix = Path(secure_filename(f.filename)).suffix.lower()
    tmp    = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        f.save(tmp.name); tmp.close()
        t0               = time.perf_counter()
        verdict, analyses = eng.analyse_video(tmp.name)
        elapsed          = round(time.perf_counter()-t0, 2)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass

    score_map   = {s.frame_index: s.fake_prob for s in verdict.frame_scores}
    frames_data = [{
        'frame_index': a.frame_index,
        'fake_prob':   round(score_map.get(a.frame_index, 0), 4),
        'has_face':    a.crop_result is not None,
        'thumbnail':   make_thumb(a, score_map.get(a.frame_index)),
    } for a in analyses]

    return jsonify({
        'verdict':          verdict.verdict,
        'weighted_score':   round(verdict.weighted_score, 4),
        'simple_score':     round(verdict.simple_score, 4),
        'confidence_tier':  verdict.confidence_tier,
        'frames_with_face': verdict.frames_with_face,
        'frames_analysed':  verdict.frames_analysed,
        'elapsed_sec':      elapsed,
        'decision_boundary':FAKE_DECISION_BOUNDARY,
        'num_frames_used':  num_frames,
        'metadata':         verdict.metadata,
        'frames':           frames_data,
    })

if __name__ == '__main__':
    get_engine(12)
    app.run(host='0.0.0.0', port=5000, debug=False)
