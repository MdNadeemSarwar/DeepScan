# DeepScan — Multi-Modal AI Media Detection System

![DeepScan](https://img.shields.io/badge/DeepScan-v2.0-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React%2FVite-61DAFB)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 About

**DeepScan** is a unified web-based forensic platform for detecting AI-generated and manipulated media across **images, videos, and audio** files. This system was developed as an M.Tech research project at **Indian Institute of Technology Patna**.

Unlike existing single-modality solutions, DeepScan integrates multiple detection techniques — deep learning models, frequency-domain forensics, metadata analysis, and watermark detection — into a single real-time web interface.

> **Author:** Md Nadeem Sarwar
> **Roll No:** 24A02RES30
> **Department:** Computer Science & Engineering
> **Institute:** Indian Institute of Technology Patna
> **Guide:** Prof. Rajiv Misra, Professor, CSE Department

---
## 🎯 Key Features

- 🖼️ **Image Detection** — XceptionNet-based face manipulation and GAN detection
- 🎥 **Video Detection** — Frame-by-frame temporal deepfake analysis using MTCNN + XceptionNet
- 🎵 **Audio Detection** — Voice cloning and synthetic speech detection using RawNet2
- 🔍 **AI Watermark Detection** — Template matching for Gemini, DALL-E, Midjourney watermarks
- 📊 **FFT Frequency Analysis** — Frequency-domain forensics for AI-generated content
- 🗂️ **Metadata Analysis** — EXIF and file metadata scanning for AI tool signatures
- 🌐 **Forensic Web Interface** — Modern dark-themed UI built with React + Vite
- ⚡ **Real-time Results** — Confidence scores with unique prediction IDs
- 📝 **Feedback System** — User feedback collection for model improvement

---

## 🏗️ System Architecture

DeepScan/
├── project/                          # Backend (FastAPI)
│   ├── main.py                       # API server & endpoints
│   ├── inference.py                  # Image/Video detection pipeline
│   ├── audio_inference.py            # Audio detection pipeline
│   ├── requirements.txt              # Python dependencies
│   └── model/
│       ├── xception_deepfake_image_5o.h5           # XceptionNet weights
│       ├── librifake_pretrained_lambda0.5_epoch_25.pth  # RawNet2 weights
│       └── gimini_logo.png           # Gemini watermark template
│
└── vision-truth-finder/              # Frontend (React + Vite)
├── index.html
└── src/
├── pages/
│   └── Index.tsx             # Main page
└── components/
├── FileUpload.tsx        # File upload component
├── ResultDisplay.tsx     # Result display component
└── FeedbackForm.tsx      # Feedback form component

## 🤖 Models & Techniques

| Component | Model/Technique | Task | Performance |
|---|---|---|---|
| Image | XceptionNet (CNN) | Face deepfake detection | ~92% accuracy |
| Video | XceptionNet + MTCNN | Frame-level deepfake detection | ~91% accuracy |
| Audio | RawNet2 | Synthetic voice detection | ~95% accuracy |
| Forensics | FFT Analysis | AI-generated content detection | Custom |
| Watermark | Template Matching | AI watermark identification | High precision |
| Metadata | EXIF Analysis | AI tool signature detection | Rule-based |

## 🔬 Detection Pipeline

### 🖼️ Image Detection:

---

## 🤖 Models & Techniques

| Component | Model/Technique | Task | Performance |
|---|---|---|---|
| Image | XceptionNet (CNN) | Face deepfake detection | ~92% accuracy |
| Video | XceptionNet + MTCNN | Frame-level deepfake detection | ~91% accuracy |
| Audio | RawNet2 | Synthetic voice detection | ~95% accuracy |
| Forensics | FFT Analysis | AI-generated content detection | Custom |
| Watermark | Template Matching | AI watermark identification | High precision |
| Metadata | EXIF Analysis | AI tool signature detection | Rule-based |

---

## 🔬 Detection Pipeline

### 🖼️ Image Detection:

Input Image
↓

Filename Analysis     → Check for AI tool names in filename
↓
Metadata Analysis     → Scan EXIF data for AI signatures
↓
Watermark Detection   → Template matching for AI watermarks
↓
FFT Analysis          → Frequency domain pattern detection
↓
XceptionNet           → Deep learning face manipulation detection
↓
Result: FAKE / AUTHENTIC + Confidence Score

### 🎥 Video Detection:

Input Video
↓

Frame Extraction      → Sample frames at configurable rate
↓
FFT Analysis          → Frequency analysis on sampled frames
↓
Face Detection        → MTCNN parallel face detection
↓
XceptionNet           → Per-face classification
↓
Ensemble Voting       → Combined score (70% CNN + 30% FFT)
↓
Result: FAKE / AUTHENTIC + Confidence Score

### 🎵 Audio Detection:

Input Audio
↓

Preprocessing         → Resample to 24kHz, pad/trim
↓
RawNet2               → Vocoder artifact detection
↓
FFT Analysis          → Frequency peak analysis
↓
Result: FAKE / AUTHENTIC + Confidence Score

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js v20+
- Git

### Step 1 — Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/deepscan.git
cd deepscan
```

### Step 2 — Backend Setup
```bash
cd project
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Download Models

**XceptionNet Model:**
- Download from [Kaggle — armanchaudhary/xception5o](https://www.kaggle.com/models/armanchaudhary/xception5o)
- Place as: `project/model/xception_deepfake_image_5o.h5`

**RawNet2 Model:**
- Download from [Google Drive](https://drive.google.com/file/d/15qOi26czvZddIbKP_SOR8SLQFZK8cf8E/view)
- Place as: `project/model/librifake_pretrained_lambda0.5_epoch_25.pth`

### Step 4 — Run Backend
```bash
cd project
python main.py
```
✅ Backend running at: `http://127.0.0.1:8000`

### Step 5 — Frontend Setup
```bash
cd vision-truth-finder
npm install
npm run dev
```
✅ Frontend running at: `http://127.0.0.1:8080`

---

## 📡 API Endpoints

| Endpoint | Method | Description | Input |
|---|---|---|---|
| `/predict` | POST | Detect fake/real media | File upload |
| `/feedback` | POST | Submit prediction feedback | JSON |
| `/health` | GET | System health check | — |
| `/metrics` | GET | Prometheus metrics | — |

### Example Response:
```json
{
  "result": "FAKE",
  "confidence": 0.9912,
  "prediction_id": "a9299166-ff33-4ffd-9085-234d89a9c736"
}
```

---

## 🧪 Supported Formats

| Media Type | Formats | Max Size |
|---|---|---|
| Image | JPEG, PNG | 10MB |
| Video | MP4 | 10MB |
| Audio | WAV, MP3, FLAC | 10MB |

---

## 📊 Research Contributions

1. **Multi-modal unified platform** — First system combining image, video, and audio deepfake detection in a single web-based forensic interface
2. **Hybrid detection pipeline** — Novel combination of deep learning (XceptionNet, RawNet2) with traditional forensic techniques (FFT, metadata analysis)
3. **AI watermark detection** — Template matching approach for identifying AI-generated content from tools like Gemini and DALL-E
4. **Frequency-domain forensics** — FFT-based analysis to detect synthetic media patterns invisible to human eye
5. **Multi-signal ensemble** — Combining multiple detection signals for improved accuracy and reduced false positives

---

## 🔮 Future Work

- Integration of larger and more diverse training datasets
- Support for additional AI generation tools (Sora, Runway, ElevenLabs)
- Mobile application deployment
- Blockchain-based media authenticity certificates
- Real-time video stream analysis

---

## 📚 References

1. Goodfellow et al., "Generative Adversarial Networks," ACM Communications, 2014
2. Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997
3. Hamza et al., "Deepfake Audio Detection via MFCC Features," IEEE Access, 2022
4. Sun et al., "AI-Synthesized Voice Detection Using Neural Vocoder Artifacts," CVPRW 2023
5. Cozzolino et al., "Forensic Analysis of Deepfake Images," IEEE Signal Processing Magazine, 2020
6. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019

---

## ⚠️ Disclaimer

This system is developed for **educational and research purposes only** at IIT Patna. It must not be used for legal judgments, surveillance, or any application requiring forensic-grade certainty without further validation. Detection accuracy varies based on media quality, compression, and AI generation method used.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>DeepScan — Multi-Modal AI Media Detection System</b><br>
  <b>Md Nadeem Sarwar | Roll No: 24A02RES30</b><br>
  Computer Science & Engineering | IIT Patna | 2024-2026<br>
  Under the guidance of Prof. Rajiv Misra
</div>
