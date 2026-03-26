Place your trained YOLOv11n model here as: yolo11n_text.pt

To get this file:
1. Follow TRAINING_GUIDE.md to train on your data
2. After training, copy: runs/detect/text_detector/weights/best.pt → models/yolo11n_text.pt

Without this file, the app uses the base YOLOv11n (not fine-tuned for text detection).
EasyOCR/Tesseract will still extract text — YOLO just won't have specialized text region detection.
