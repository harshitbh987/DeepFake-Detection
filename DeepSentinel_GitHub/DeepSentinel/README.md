# 🔍 DeepSentinel — Deepfake Video Detection System

A production-ready deepfake video detection system using **EfficientNet-B0** and **MTCNN**, deployable on any local machine (Windows / Mac / Linux).

> **Stack:** Python 3.10 · PyTorch · EfficientNet-B0 · MTCNN · OpenCV · GradCAM · Streamlit · Flask

---

## 📁 Project Structure

```
DeepSentinel/
├── config.py                    # Central config — all paths & hyperparameters
├── train.py                     # Train the model (Phase 1 warm-up + Phase 2 fine-tune)
├── predict.py                   # Predict on a single video (CLI)
├── plot_curves.py               # Plot training loss & AUC curves
├── run_streamlit.py             # Launch Streamlit web interface
├── run_flask.py                 # Launch Flask web interface
├── app.py                       # Streamlit app source
├── flask_app.py                 # Flask backend
├── requirements.txt
│
├── models/
│   └── model_loader.py          # SentinelNet (EfficientNet-B0 + GELU head)
├── detector/
│   ├── inference_engine.py      # Full pipeline: extract → detect → classify → aggregate
│   ├── face_cropper.py          # MTCNN face detection & alignment
│   └── frame_sampler.py         # OpenCV frame extraction
├── training/
│   └── dataset_builder.py       # Dataset loading & DataLoaders
├── utils/
│   ├── video_reader.py          # Video utilities
│   └── score_manager.py         # Score tracking & export
├── visualization/
│   └── confidence_plot.py       # GradCAM & plot utilities
├── templates/
│   └── index.html               # Flask HTML/CSS/JS frontend
│
├── dataset/                     # ← Put your videos here (gitignored)
│   ├── real/                    #     Genuine video files
│   └── fake/                    #     Deepfake video files
└── checkpoints/                 # ← Model weights saved here (gitignored)
    └── sentinel_b0_best.pt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/harshitbh987/DeepSentinel.git
cd DeepSentinel
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your videos in the dataset folder:
```
dataset/
├── real/   ← genuine MP4/AVI/MOV files
└── fake/   ← deepfake MP4/AVI/MOV files
```

### 3. Train the Model

```bash
python train.py
# With options:
python train.py --warmup_epochs 5 --finetune_epochs 20 --device cuda
```

### 4. Plot Training Curves

```bash
python plot_curves.py
# Output: training_curves.png
```

### 5. Run Inference on a Video

```bash
python predict.py --video path/to/video.mp4
python predict.py --video video.mp4 --plot       # save score chart
python predict.py --video video.mp4 --gradcam    # GradCAM heatmaps
```

### 6. Launch Web Interface

**Streamlit (recommended):**
```bash
python run_streamlit.py
# Open http://localhost:8501
```

**Flask (professional UI):**
```bash
python run_flask.py
# Open http://localhost:5000
```

---

## 🧠 Model Architecture

| Component          | Detail                                      |
|--------------------|---------------------------------------------|
| Backbone           | EfficientNet-B0 (pre-trained on ImageNet)   |
| Classification Head| Dropout(0.4) → Linear(1280→256) → GELU → Dropout(0.2) → Linear(256→1) |
| Loss Function      | Label-smoothed BCE (ε=0.05)                 |
| Optimizer          | AdamW + CosineAnnealingLR                   |
| Face Detector      | MTCNN (facenet-pytorch)                     |
| Interpretability   | GradCAM saliency maps                       |

---

## 📊 Results

| Metric      | Value  |
|-------------|--------|
| AUC-ROC     | 0.921  |
| F1-Score    | 0.875  |
| Accuracy    | 86.4%  |
| Inference   | ~9ms/frame (GPU) |

---

## ⚙️ Configuration

Edit `config.py` to change:
- Dataset path (`DATASET_ROOT`)
- Training hyperparameters (`WARMUP_LR`, `FINETUNE_LR`, `BATCH_SIZE`, etc.)
- Detection threshold (`FAKE_DECISION_BOUNDARY`)
- Model checkpoint path (`PRIMARY_WEIGHTS`)

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named cv2` | `pip install opencv-python` |
| `ModuleNotFoundError: numpy.char` | `pip install numpy==1.26.4 --force-reinstall` |
| `UnpicklingError: weights_only` | Already patched in all launchers |
| `ImportError: DATASET_ROOT` | Replace `config.py` with the one in this repo |
| `FileNotFoundError: checkpoint` | Run `train.py` first |
| CUDA out of memory | Reduce `BATCH_SIZE` in `config.py` |
| Streamlit PATH error | Use `python run_streamlit.py` not `streamlit run app.py` directly |

---

## 📄 License

MIT License — free for academic and research use.

---

## 👤 Author

**Harshit Bhadouriya**  
B.Tech CSE (AI/ML) — NIET Greater Noida, Batch 2026  
[LinkedIn](https://linkedin.com/in/harshit-bhadouriya) · [GitHub](https://github.com/harshitbh987)
