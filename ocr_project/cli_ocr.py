"""
cli_ocr.py — Command-Line OCR Tool
===================================
Usage:
  python cli_ocr.py --input image.jpg
  python cli_ocr.py --input folder/ --batch
  python cli_ocr.py --input image.jpg --output result.json --annotate
"""

import argparse
import json
import os
import sys
from pathlib import Path
from ocr_engine import OCREngine


SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def process_single(engine: OCREngine, input_path: str, output_dir: str, annotate: bool = True) -> dict:
    stem = Path(input_path).stem
    annotated_path = os.path.join(output_dir, f'{stem}_annotated.jpg') if annotate else None
    result = engine.process_image(input_path, annotated_path or '/tmp/_tmp_annotated.jpg')
    return result


def print_result(result: dict, input_path: str):
    print(f"\n{'═'*60}")
    print(f"  File : {input_path}")
    print(f"  Regions detected: {result['total_regions']}")
    print(f"{'─'*60}")
    for region in result['regions']:
        print(f"\n  📍 Region {region['id']}  (confidence: {region['confidence']:.2f})")
        print(f"     Bounding box: {region['bbox']}")
        print(f"     Text: {region['text']}")
    print(f"\n{'─'*60}")
    print(f"  📝 FULL EXTRACTED TEXT:\n")
    print(result['full_text'])
    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='OCR Text Extractor with YOLOv11n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_ocr.py --input photo.jpg
  python cli_ocr.py --input docs/ --batch --output results/
  python cli_ocr.py --input scan.png --json output.json
        """
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input image file or directory (for batch)')
    parser.add_argument('--output', '-o', default='outputs/',
                        help='Output directory for annotated images')
    parser.add_argument('--json', '-j', default=None,
                        help='Save results as JSON to this file')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all images in input directory')
    parser.add_argument('--no-annotate', action='store_true',
                        help='Skip saving annotated images')
    parser.add_argument('--model', '-m', default='models/yolo11n_text.pt',
                        help='Path to YOLO model weights')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("🔄 Loading OCR engine...")
    engine = OCREngine(yolo_model_path=args.model)
    print(f"   OCR backend: {engine.ocr_backend}")
    print(f"   YOLO loaded: {engine.yolo_loaded}")

    all_results = []

    if args.batch or os.path.isdir(args.input):
        # Batch mode
        image_files = [
            f for f in Path(args.input).iterdir()
            if f.suffix.lower() in SUPPORTED
        ]
        print(f"\n📂 Found {len(image_files)} images in {args.input}")

        for i, img_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing: {img_path.name}")
            try:
                result = process_single(engine, str(img_path), args.output, not args.no_annotate)
                result['file'] = str(img_path)
                all_results.append(result)
                print_result(result, str(img_path))
            except Exception as e:
                print(f"  ❌ Error: {e}")
    else:
        # Single file
        result = process_single(engine, args.input, args.output, not args.no_annotate)
        result['file'] = args.input
        all_results.append(result)
        print_result(result, args.input)

    # Save JSON if requested
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to {args.json}")

    print(f"\n✅ Done! Processed {len(all_results)} image(s).")
    if not args.no_annotate:
        print(f"   Annotated images saved to: {args.output}")


if __name__ == '__main__':
    main()
