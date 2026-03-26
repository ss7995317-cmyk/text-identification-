# 🔍 OCR Studio — YOLOv11n Text Detection

A complete OCR (Optical Character Recognition) system combining **YOLOv11n** for text region detection
and **EasyOCR** for text extraction — with a web UI and CLI tool.

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
cd ocr_project
pip install -r requirements.txt
```

> **Note:** On first run, EasyOCR downloads ~200MB of language models automatically.  
> YOLOv11n base weights (~6MB) also download automatically on first run.

### 2. Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.  
Upload any image → click **Run OCR** → see extracted text + annotated image.

### 3. Use the CLI Tool

```bash
# Single image
python cli_ocr.py --input photo.jpg

# Batch process a folder
python cli_ocr.py --input scans/ --batch

# Save results to JSON
python cli_ocr.py --input document.png --json results.json
```

---

## 🏗️ How It Works

```
Input Image
    ↓
Preprocessing (denoise, adaptive threshold, resize)
    ↓
YOLOv11n (detect text bounding boxes)
    ↓
EasyOCR / Tesseract (read text in each region)
    ↓
Annotated Image + Extracted Text
```

---

## 📁 Project Structure

```
ocr_project/
├── app.py              ← Flask web server (start here)
├── ocr_engine.py       ← Core OCR + YOLO pipeline
├── cli_ocr.py          ← Command-line tool
├── train_yolo.py       ← Training script
├── requirements.txt    ← Dependencies
├── TRAINING_GUIDE.md   ← Full training instructions
├── models/             ← Place trained .pt model here
├── training/           ← Training data & labels
├── templates/          ← Web UI
├── uploads/            ← Temp storage (auto-created)
└── outputs/            ← Annotated images (auto-created)
```

---

## 🏋️ Training on New Text

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for the full step-by-step guide.

Quick summary:
1. Place images in `training/images/train/` and `training/images/val/`
2. Label with LabelImg (YOLO format) → save to `training/labels/`
3. Edit `training/dataset.yaml` with your class names
4. Run `python train_yolo.py`
5. Model auto-saved to `models/yolo11n_text.pt`

---

## 🖥️ API Reference

### POST /api/ocr

Upload an image for OCR processing.

```bash
curl -X POST http://localhost:5000/api/ocr \
  -F "file=@document.jpg"
```

**Response:**
```json
{
  "success": true,
  "total_regions": 3,
  "full_text": "Hello World\nThis is a sample document.",
  "regions": [
    {
      "id": 1,
      "text": "Hello World",
      "bbox": [45, 30, 280, 65],
      "confidence": 0.94
    }
  ],
  "image_size": {"width": 800, "height": 600},
  "annotated_image_url": "/outputs/abc123_annotated.jpg"
}
```

### GET /api/health
Check engine status (YOLO loaded, OCR backend).

---

## 🛠️ Requirements

- Python 3.9+
- At least one OCR backend: **EasyOCR** (recommended) or **Tesseract**
- ~1GB disk space for model downloads
- GPU optional but speeds up training significantly
