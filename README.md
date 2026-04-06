# ⚡ NeuroLab v3.0 — Neural Network Toolbox

A production-grade interactive neural network laboratory with a stunning 3D UI,
built in Python with Streamlit.

## Features

| Module | Description |
|--------|-------------|
| 🧠 Perceptron | Train AND/OR/NAND/NOR/XOR gates with real-time loss curves |
| ➡️ Forward Pass | Inspect layer-by-layer activations with color heatmaps |
| ⬅️ Backprop | Train 2-layer MLP on XOR, visualize gradient descent |
| 👁️ Vision Hub | **Live webcam** — face detection, face mesh, pose estimation, edge detection, color analysis, motion detection |
| 📝 Sentiment | Train bag-of-words classifier on custom text, real-time inference |

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## Vision Hub Modes

- **Face Detection** — Haar cascade bounding boxes + eye detection
- **Face Mesh** — 468 MediaPipe facial landmarks
- **Pose Estimation** — 33-point full-body skeleton
- **Edge Detection (Canny)** — Adjustable threshold edge highlighting
- **Color Analysis** — Real-time RGB histogram bars
- **Motion Detection** — Frame-difference motion bounding boxes

## Requirements

- Python 3.9+
- Webcam (for Vision Hub)
- All packages in requirements.txt
