"""
train_yolo.py — Fine-tune YOLOv11n for Custom Text Detection
=============================================================

STEPS TO TRAIN YOUR OWN TEXT DETECTION MODEL:
----------------------------------------------
1. Prepare images  → Place in training/images/train/ and training/images/val/
2. Label images    → Use LabelImg or Roboflow, save YOLO format labels
3. Edit dataset.yaml with your class names
4. Run this script → python train_yolo.py
5. Best model saved → runs/detect/text_detector/weights/best.pt
6. Copy to models/ → cp runs/detect/text_detector/weights/best.pt models/yolo11n_text.pt
"""

import os
import yaml
import shutil
from pathlib import Path


# ─────────────────────────────── CONFIG ─────────────────────────────────────

TRAINING_CONFIG = {
    'model': 'yolo11n.pt',          # Base model (downloads automatically)
    'epochs': 100,                   # Increase for better accuracy (200-300 recommended)
    'batch': 16,                     # Reduce to 8 if GPU memory error
    'imgsz': 640,                    # Input image size
    'lr0': 0.01,                     # Initial learning rate
    'patience': 20,                  # Early stopping patience
    'workers': 4,                    # Dataloader workers
    'device': '',                    # '' = auto (GPU if available, else CPU)
    'project': 'runs/detect',
    'name': 'text_detector',
    'save_period': 10,               # Save checkpoint every N epochs
    'augment': True,                 # Data augmentation
    'mosaic': 1.0,                   # Mosaic augmentation
    'degrees': 5.0,                  # Rotation augmentation
    'translate': 0.1,
    'scale': 0.5,
    'flipud': 0.0,                   # No vertical flip for text
    'fliplr': 0.0,                   # No horizontal flip for text (breaks reading direction)
}


# ─────────────────────────────── DATASET YAML ────────────────────────────────

def create_dataset_yaml(classes: list = None):
    """Create dataset.yaml for YOLO training."""
    if classes is None:
        classes = ['text']  # Default: single 'text' class

    dataset = {
        'path': str(Path('training').absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    yaml_path = 'training/dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Dataset config saved to {yaml_path}")
    print(f"   Classes ({len(classes)}): {classes}")
    return yaml_path


# ─────────────────────────────── LABEL CONVERTER ─────────────────────────────

def convert_labelimg_to_yolo(annotation_dir: str, output_dir: str):
    """
    Convert LabelImg XML annotations to YOLO format.
    Use this if you labeled with LabelImg in PascalVOC format.
    """
    import xml.etree.ElementTree as ET
    import glob

    xml_files = glob.glob(os.path.join(annotation_dir, '*.xml'))
    os.makedirs(output_dir, exist_ok=True)

    class_map = {}
    class_counter = 0

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_w = int(root.find('size/width').text)
        img_h = int(root.find('size/height').text)

        label_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_map:
                class_map[class_name] = class_counter
                class_counter += 1

            class_id = class_map[class_name]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert to YOLO format (normalized cx, cy, w, h)
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h

            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        base_name = Path(xml_file).stem
        label_file = os.path.join(output_dir, f'{base_name}.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(label_lines))

    print(f"✅ Converted {len(xml_files)} XML annotations to YOLO format")
    print(f"   Class map: {class_map}")
    return class_map


# ─────────────────────────────── VALIDATE DATASET ────────────────────────────

def validate_dataset():
    """Check that dataset structure is correct before training."""
    required = [
        'training/images/train',
        'training/images/val',
        'training/labels/train',
        'training/labels/val',
    ]

    print("\n📋 Validating dataset structure...")
    all_ok = True

    for path in required:
        files = list(Path(path).glob('*')) if Path(path).exists() else []
        status = '✅' if files else '❌ EMPTY'
        print(f"  {status} {path}/ — {len(files)} files")
        if not files:
            all_ok = False

    # Check label-image pairing
    train_images = set(p.stem for p in Path('training/images/train').glob('*'))
    train_labels = set(p.stem for p in Path('training/labels/train').glob('*.txt'))
    missing_labels = train_images - train_labels

    if missing_labels:
        print(f"\n⚠️  {len(missing_labels)} images missing labels: {list(missing_labels)[:5]}")
        all_ok = False
    else:
        print(f"\n✅ All {len(train_images)} training images have labels")

    return all_ok


# ─────────────────────────────── TRAIN ───────────────────────────────────────

def train(classes: list = None, extra_config: dict = None):
    """
    Main training function.

    Args:
        classes: List of class names, e.g. ['printed_text', 'handwritten_text']
        extra_config: Override any TRAINING_CONFIG values
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return

    if classes is None:
        classes = ['text']

    # Validate
    if not validate_dataset():
        print("\n❌ Dataset validation failed. Please add images and labels first.")
        print("   See TRAINING_GUIDE.md for step-by-step instructions.")
        return

    # Create dataset YAML
    yaml_path = create_dataset_yaml(classes)

    # Load base model
    config = {**TRAINING_CONFIG}
    if extra_config:
        config.update(extra_config)

    print(f"\n🚀 Starting YOLOv11n training...")
    print(f"   Model: {config['model']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Classes: {classes}")
    print(f"   Device: {'auto' if not config['device'] else config['device']}\n")

    model = YOLO(config['model'])

    results = model.train(
        data=yaml_path,
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        lr0=config['lr0'],
        patience=config['patience'],
        workers=config['workers'],
        device=config['device'],
        project=config['project'],
        name=config['name'],
        save_period=config['save_period'],
        augment=config['augment'],
        mosaic=config['mosaic'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        flipud=config['flipud'],
        fliplr=config['fliplr'],
        exist_ok=True
    )

    best_model = f"runs/detect/{config['name']}/weights/best.pt"
    if os.path.exists(best_model):
        os.makedirs('models', exist_ok=True)
        shutil.copy(best_model, 'models/yolo11n_text.pt')
        print(f"\n✅ Training complete! Best model saved to models/yolo11n_text.pt")
        print(f"   Restart the app to use the new model.")
    else:
        print(f"\n⚠️  Training done but best.pt not found at {best_model}")

    return results


# ─────────────────────────────── EXPORT ──────────────────────────────────────

def export_model(format: str = 'onnx'):
    """
    Export trained model to ONNX/TFLite/CoreML for deployment.

    Args:
        format: 'onnx', 'tflite', 'coreml', 'torchscript'
    """
    from ultralytics import YOLO
    model_path = 'models/yolo11n_text.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Train first.")
        return

    model = YOLO(model_path)
    export_path = model.export(format=format)
    print(f"✅ Model exported to: {export_path}")


# ─────────────────────────────── ENTRY POINT ─────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv11n for text detection')
    parser.add_argument('--classes', nargs='+', default=['text'],
                        help='Class names, e.g. --classes printed_text handwritten_text')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='',
                        help='Device: "" (auto), "0" (GPU 0), "cpu"')
    parser.add_argument('--export', type=str, default=None,
                        help='Export format after training: onnx, tflite, coreml')
    args = parser.parse_args()

    train(
        classes=args.classes,
        extra_config={
            'epochs': args.epochs,
            'batch': args.batch,
            'device': args.device
        }
    )

    if args.export:
        export_model(args.export)
