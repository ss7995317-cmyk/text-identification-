"""
OCR Project with YOLOv11n Text Detection
Flask Web Application - Main Entry Point
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid
from pathlib import Path
from ocr_engine import OCREngine

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Initialize OCR Engine (loads YOLO + Tesseract/EasyOCR)
ocr_engine = OCREngine()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported: {list(ALLOWED_EXTENSIONS)}'}), 400

    file_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[1].lower()
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.{ext}')
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{file_id}_annotated.jpg')

    file.save(input_path)

    try:
        result = ocr_engine.process_image(input_path, output_path)
        result['file_id'] = file_id
        result['annotated_image_url'] = f'/outputs/{file_id}_annotated.jpg'
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'yolo_loaded': ocr_engine.yolo_loaded,
        'ocr_backend': ocr_engine.ocr_backend
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    print("🚀 OCR Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
