# 🔍 AI Threat Detection System

An AI-powered real-time threat detection system using YOLOv8 computer vision to enhance security in public spaces.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)

---

## 🎯 Overview

This system uses deep learning to detect threats in images, videos, and real-time webcam feeds. It's designed to integrate with existing security infrastructure and provide instant alerts when threats are detected.

**Key Use Cases:**
- Public space security monitoring
- CCTV camera integration
- Real-time threat alerting
- Security personnel assistance

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Real-time Webcam Detection** | Live threat detection using your webcam with FPS counter |
| 🖼️ **Image Analysis** | Upload and analyze images for threat detection |
| 🎬 **Video Processing** | Process video files with frame-by-frame analysis |
| 🎨 **Gamma Correction** | Automatic brightness adjustment for low-light conditions |
| 📊 **Detection Statistics** | Detailed statistics and confidence scores |
| 🌐 **Web Interface** | Modern, responsive web UI |

---

## 💻 Requirements

### System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **Webcam:** Required for real-time detection feature

### Hardware Options
- **CPU Mode:** Works on any modern CPU (slower detection)
- **GPU Mode:** NVIDIA GPU with CUDA support (faster detection)

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
git clone <https://github.com/ahmetsezginn/Weapon-detection>
cd weapon-detection/web
```

Or download and extract the ZIP file.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

#### Option A: CPU Installation (Default)
```bash
pip install -r requirements.txt
```

#### Option B: GPU Installation (NVIDIA CUDA)

First, check your CUDA version:
```bash
nvidia-smi
```

Then install the appropriate requirements:

```bash
# For CUDA 11.8 (GTX 10/16/20/30 series, RTX 20/30 series)
pip install -r requirements-gpu.txt

# For CUDA 12.1 (RTX 40 series and newer)
pip install -r requirements-gpu-cu121.txt
```

#### Verify GPU Installation
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Requirements Files Summary

| File | CUDA Version | Recommended For |
|------|--------------|-----------------|
| `requirements.txt` | None (CPU) | No GPU / AMD GPU / Testing |
| `requirements-gpu.txt` | CUDA 11.8 | GTX 1000/1600, RTX 2000/3000 series |
| `requirements-gpu-cu121.txt` | CUDA 12.1 | RTX 4000 series and newer |

---

## ▶️ Running the Application

### Step 1: Activate Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 2: Verify Model File

Make sure `best.pt` (the trained YOLO model) exists in the project directory:
```bash
# Windows
dir best.pt

# macOS/Linux
ls -la best.pt
```

### Step 3: Start the Server

```bash
python app.py
```

You should see output like:
```
==================================================
Starting Flask Threat Detection System...
==================================================

Loading model from: best.pt
Model loaded successfully
Model classes: {0: 'threat', 1: 'no-threat'}
Model loaded successfully!

Server starting on http://localhost:5000
==================================================

 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

---

## 📖 Usage Guide

### 🏠 Home Page
The landing page with an overview of the system and its features.

### 🎥 Real-time Camera Detection

1. Go to **Demo** page
2. Click **"Real-time Camera Detection"**
3. Click **"Start Camera"** button
4. Allow camera access when prompted
5. The system will detect threats in real-time
6. Threats are highlighted with **red** bounding boxes
7. Non-threats are highlighted with **green** bounding boxes
8. Click **"Stop Camera"** when done

**Keyboard Shortcut:** Press `Q` to stop detection (when camera window is focused)

### 🖼️ Image Detection

1. Go to **Demo** page
2. Click **"Still Image Detection"**
3. Drag & drop an image or click to browse
4. The system will analyze and show results
5. Detected objects are highlighted with bounding boxes

**Supported formats:** JPG, JPEG, PNG

### 🎬 Video Detection

1. Go to **Demo** page
2. Click **"Video Detection"**
3. Upload a video file
4. Wait for processing (progress shown)
5. View the processed video with detections
6. See frame-by-frame detection statistics

**Supported formats:** MP4, AVI, MOV, WEBM

---

## 📁 Project Structure

```
weapon-detection/web/
│
├── app.py                    # Main Flask application
├── video_processor.py        # Video processing module
├── best.pt                   # Trained YOLOv8 model
│
├── requirements.txt          # CPU dependencies
├── requirements-gpu.txt      # GPU dependencies (CUDA 11.8)
├── requirements-gpu-cu121.txt # GPU dependencies (CUDA 12.1)
│
├── templates/                # HTML templates
│   ├── base.html             # Base template with navigation
│   ├── index.html            # Home page
│   ├── overview.html         # Project overview
│   ├── architecture.html     # System architecture
│   ├── demo.html             # Live demo page
│   ├── team.html             # Team information
│   └── contact.html          # Contact and references
│
├── static/                   # Static assets
│   ├── css/
│   │   └── style.css         # Custom styles
│   ├── js/
│   │   └── script.js         # Client-side JavaScript
│   ├── images/               # Image assets
│   └── videos/               # Video assets
│
├── uploads/                  # Uploaded files (auto-created)
├── runs/                     # YOLO output directory
└── .venv/                    # Virtual environment (auto-created)
```

---

## 🔧 Technologies Used

| Category | Technology |
|----------|------------|
| **Backend** | Python 3.10, Flask 2.3 |
| **AI Model** | YOLOv8 (Ultralytics) |
| **Computer Vision** | OpenCV 4.8, Pillow |
| **Deep Learning** | PyTorch |
| **Video Processing** | imageio-ffmpeg |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5 |

---

## 🐛 Troubleshooting

### Camera Not Working
- Make sure no other application is using the camera
- Check camera permissions in your browser/OS
- Try a different browser (Chrome recommended)

### Model Not Loading
- Verify `best.pt` file exists in the project directory
- Check file size (should be several MB)
- Re-download the model if corrupted

### Slow Detection
- Use GPU version for faster detection
- Close other resource-intensive applications
- Reduce video resolution

### Video Not Playing in Browser
- The system automatically converts videos to browser-compatible format
- If still not working, try a different browser
- Check that imageio-ffmpeg is installed

---

## 📄 License

This project is for educational purposes only.

---

## 👥 Contributors

**Hasan Kalyoncu University** - Faculty of Engineering, Computer Engineering Department

### Project Team

| Name | Role |
|------|------|
| **Ahmet Sezgin** | Developer |
| **Amani Kanti** | Developer |
| **Nejdet Çevik** | Developer |

### Supervisor

| Name | Title |
|------|-------|
| **Saed ALQARALEH** | Asst. Prof. Dr. - Project Advisor |

---

This project was developed as part of an academic course on AI and Computer Vision at Hasan Kalyoncu University.
