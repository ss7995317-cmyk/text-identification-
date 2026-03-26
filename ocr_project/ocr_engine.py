"""
OCR Engine - Core Processing Module
Combines YOLOv11n for text region detection + EasyOCR/Tesseract for text extraction
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path


class OCREngine:
    def __init__(self, yolo_model_path: str = "models/yolo11n_text.pt"):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.yolo_loaded = False
        self.ocr_backend = None
        self.reader = None

        self._load_yolo()
        self._load_ocr_backend()

    # ─────────────────────────────── LOADERS ────────────────────────────────

    def _load_yolo(self):
        """Load YOLOv11n model for text region detection."""
        try:
            from ultralytics import YOLO
            if os.path.exists(self.yolo_model_path):
                self.yolo_model = YOLO(self.yolo_model_path)
                print(f"✅ YOLO model loaded from {self.yolo_model_path}")
            else:
                # Download pretrained YOLOv11n as base (fine-tune for text later)
                self.yolo_model = YOLO("yolo11n.pt")
                os.makedirs("models", exist_ok=True)
                print("✅ YOLOv11n base model loaded (not yet fine-tuned for text)")
            self.yolo_loaded = True
        except ImportError:
            print("⚠️  ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            print(f"⚠️  YOLO load error: {e}")

    def _load_ocr_backend(self):
        """Try EasyOCR first, fall back to Tesseract, then pure OpenCV."""
        # Try EasyOCR
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.ocr_backend = 'easyocr'
            print("✅ OCR backend: EasyOCR")
            return
        except ImportError:
            pass

        # Try Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.ocr_backend = 'tesseract'
            print("✅ OCR backend: Tesseract")
            return
        except Exception:
            pass

        # Fallback: OpenCV + basic blob detection (limited accuracy)
        self.ocr_backend = 'opencv_fallback'
        print("⚠️  OCR backend: OpenCV fallback (install EasyOCR or Tesseract for best results)")

    # ─────────────────────────────── MAIN PIPELINE ──────────────────────────

    def process_image(self, input_path: str, output_path: str) -> dict:
        """
        Full OCR pipeline:
        1. Preprocess image
        2. Detect text regions with YOLO (or fallback contour detection)
        3. Run OCR on each region
        4. Annotate output image
        5. Return structured results
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")

        original = image.copy()
        preprocessed = self._preprocess(image)

        # Step 1: Detect text bounding boxes
        regions = self._detect_text_regions(preprocessed, original)

        # Step 2: Run OCR on each region
        extracted_texts = []
        annotated = original.copy()

        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region['bbox']
            # Crop with small padding
            pad = 4
            crop = original[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]
            if crop.size == 0:
                continue

            text = self._run_ocr(crop)
            if text.strip():
                extracted_texts.append({
                    'id': i + 1,
                    'text': text.strip(),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': region.get('confidence', 1.0)
                })
                self._draw_annotation(annotated, x1, y1, x2, y2, text, i + 1)

        # If YOLO/contours found nothing useful, run full-image OCR
        if not extracted_texts:
            full_text = self._run_ocr(original)
            if full_text.strip():
                extracted_texts.append({
                    'id': 1,
                    'text': full_text.strip(),
                    'bbox': [0, 0, original.shape[1], original.shape[0]],
                    'confidence': 1.0
                })

        cv2.imwrite(output_path, annotated)

        combined_text = '\n'.join([t['text'] for t in extracted_texts])
        return {
            'success': True,
            'total_regions': len(extracted_texts),
            'full_text': combined_text,
            'regions': extracted_texts,
            'image_size': {'width': original.shape[1], 'height': original.shape[0]}
        }

    # ─────────────────────────────── PREPROCESSING ──────────────────────────

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR accuracy."""
        # Resize if too small
        h, w = image.shape[:2]
        if max(h, w) < 800:
            scale = 800 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    # ─────────────────────────────── TEXT DETECTION ─────────────────────────

    def _detect_text_regions(self, preprocessed: np.ndarray, original: np.ndarray) -> list:
        """Detect text regions using YOLO or contour fallback."""
        if self.yolo_loaded and self.yolo_model is not None:
            return self._detect_with_yolo(original)
        return self._detect_with_contours(preprocessed, original)

    def _detect_with_yolo(self, image: np.ndarray) -> list:
        """Use YOLOv11n to detect text regions."""
        results = self.yolo_model(image, verbose=False)
        regions = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.25:
                    regions.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})

        # If YOLO finds nothing (not fine-tuned for text yet), fall back
        if not regions:
            regions = self._detect_with_contours(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), image
            )
        return regions

    def _detect_with_contours(self, gray: np.ndarray, original: np.ndarray) -> list:
        """Fallback: detect text regions via MSER + contour analysis."""
        h, w = original.shape[:2]

        # MSER for text blob detection
        mser = cv2.MSER_create()
        try:
            if len(gray.shape) == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            regions_mser, _ = mser.detectRegions(gray)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions_mser]
        except Exception:
            hulls = []

        # Merge nearby bounding boxes
        bboxes = []
        for hull in hulls:
            rect = cv2.boundingRect(hull)
            x, y, rw, rh = rect
            aspect = rw / max(rh, 1)
            # Filter: text regions are usually wider than tall, not too tiny
            if 0.1 < aspect < 15 and rw > 10 and rh > 8:
                bboxes.append([x, y, x + rw, y + rh])

        merged = self._merge_boxes(bboxes, merge_threshold=12)

        # Clamp to image bounds
        regions = []
        for box in merged:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 10 and (y2 - y1) > 8:
                regions.append({'bbox': [x1, y1, x2, y2], 'confidence': 0.8})

        # If still empty, return full image as one region
        if not regions:
            regions = [{'bbox': [0, 0, w, h], 'confidence': 1.0}]

        return regions

    def _merge_boxes(self, boxes: list, merge_threshold: int = 10) -> list:
        """Merge overlapping/nearby bounding boxes."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = [boxes[0]]

        for box in boxes[1:]:
            last = merged[-1]
            # Check overlap or proximity
            if (box[0] < last[2] + merge_threshold and
                    box[1] < last[3] + merge_threshold and
                    box[2] > last[0] - merge_threshold and
                    box[3] > last[1] - merge_threshold):
                merged[-1] = [
                    min(last[0], box[0]),
                    min(last[1], box[1]),
                    max(last[2], box[2]),
                    max(last[3], box[3])
                ]
            else:
                merged.append(box)

        return merged

    # ─────────────────────────────── OCR BACKENDS ───────────────────────────

    def _run_ocr(self, crop: np.ndarray) -> str:
        """Run OCR on a cropped region using the best available backend."""
        if self.ocr_backend == 'easyocr':
            return self._ocr_easyocr(crop)
        elif self.ocr_backend == 'tesseract':
            return self._ocr_tesseract(crop)
        else:
            return "[Install EasyOCR or Tesseract for text extraction]"

    def _ocr_easyocr(self, crop: np.ndarray) -> str:
        try:
            results = self.reader.readtext(crop, detail=0, paragraph=True)
            return ' '.join(results)
        except Exception as e:
            return f"[EasyOCR error: {e}]"

    def _ocr_tesseract(self, crop: np.ndarray) -> str:
        try:
            import pytesseract
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            config = '--oem 3 --psm 6'
            return pytesseract.image_to_string(gray, config=config)
        except Exception as e:
            return f"[Tesseract error: {e}]"

    # ─────────────────────────────── ANNOTATION ─────────────────────────────

    def _draw_annotation(self, image, x1, y1, x2, y2, text, region_id):
        """Draw bounding box + label on the annotated image."""
        # Color palette for different regions
        colors = [
            (0, 200, 100), (0, 120, 255), (255, 80, 0),
            (180, 0, 255), (0, 200, 200), (255, 180, 0)
        ]
        color = colors[(region_id - 1) % len(colors)]

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Truncate long text for label
        label_text = text[:40] + '...' if len(text) > 40 else text
        label = f"[{region_id}] {label_text}"

        # Background for label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(image, (x1, label_y - th - 4), (x1 + tw + 4, label_y + baseline), color, -1)
        cv2.putText(image, label, (x1 + 2, label_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
