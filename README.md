---

# 🚗 Global ALPR System 🌍

### Real-Time License Plate Recognition (ALPR)

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Used-orange?style=for-the-badge&logo=pytorch" />
  <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Detection-8A2BE2?style=for-the-badge" />
  <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

## 📌 Overview

**Global ALPR System** is a real-time Automatic License Plate Recognition application designed using:

* **YOLOv8** — fast vehicle & plate detection
* **DeepSORT** — stable object tracking with persistent IDs
* **EasyOCR** — multi-regional license plate OCR
* **Streamlit** — intuitive, modern web interface

It processes uploaded videos and outputs:

✔ Vehicle detection
✔ Plate extraction
✔ Tracking with stable IDs
✔ Clean HUD overlays
✔ OCR text stabilization using majority voting

---

## 🌐 Live Demo

**[https://ai-license-plate-recognition-system-4vlwziutnygjva9rjqq9yp.streamlit.app](https://ai-license-plate-recognition-system-4vlwziutnygjva9rjqq9yp.streamlit.app)**

---

## 🌟 Features

### 🔍 YOLOv8-Based Detection

* Accurate plate and vehicle detection
* TorchScript export for optimized runtime

### 🎯 DeepSORT Tracking

* Smooth, flicker-free tracking
* Unique IDs for each vehicle

### 🔤 Global OCR Support

* EasyOCR for international license plates

### 🧠 Text Stabilization

* Majority-vote system for stable OCR output

### 🎨 Clean HUD Overlay

* Floating labels
* Enhanced plate cropping
* Enlarged preview

### ⚡ Hardware Acceleration

* Runs on CUDA, Apple MPS, or CPU

### 🖥 Streamlit UI

* Dark theme
* Simple drag-and-drop uploader
* Real-time progress

---

## 🧰 Tech Stack

| Component | Technology                     |
| --------- | ------------------------------ |
| Detection | YOLOv8                         |
| Tracking  | DeepSORT                       |
| OCR       | EasyOCR                        |
| Backend   | Python 3.10                    |
| UI        | Streamlit                      |
| Libraries | PyTorch, OpenCV, NumPy, Pandas |

---

## 📁 Project Structure

Your repository contains only essential files (clean and GitHub-friendly):

```
ALPR-SYSTEM/
├── models/             
├── detector.py           
├── streamlit_app.py      
├── utils.py             
├── train.py             
├── requirements.txt
└── README.md
```

**Not included in repo**: `__pycache__`, `Outputs`, `videos`, temporary files.

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ALPR-SYSTEM.git
cd ALPR-SYSTEM
```

### 2️⃣ Create & activate environment

```bash
conda create -n alpr_env python=3.10
conda activate alpr_env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add YOLO model

Place your **TorchScript** model here:

```
models/best.torchscript
```

To export your YOLOv8 model:

```bash
yolo export model=best.pt format=torchscript
```

---

## ▶️ Usage

### Run the application:

```bash
conda activate alpr_env
streamlit run streamlit_app.py
```

Then open:

```
http://localhost:8501
```

Upload a video → the system processes it → annotated result is generated locally.

---

## 🧠 Model Training

To train YOLOv8 (example for Roboflow dataset):

```bash
python train.py --api-key YOUR_ROBOFLOW_KEY --epochs 50 --project training_runs
```

---

