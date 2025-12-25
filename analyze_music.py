#!/usr/bin/env python3
"""
Music Scanner - Analyze sheet music and generate three output versions
Usage: python3 analyze_music.py <input_image>
"""

import sys
import cv2
import numpy as np
from collections import Counter
from scanner.detection import detect_everything, visualize_predictions

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_music.py <input_image>")
        print("Example: python3 analyze_music.py input.png")
        sys.exit(1)

    input_path = sys.argv[1]

    # Load image
    print(f"Loading image: {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image from {input_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

    # Run detection
    print("\nRunning detection...")
    result = detect_everything(image_rgb)
    print(f"✓ Detected {len(result.object_prediction_list)} total objects")

    # Show all detected categories
    all_categories = Counter([pred.category.name for pred in result.object_prediction_list])
    print("\nAll detected objects:")
    for cat, count in sorted(all_categories.items()):
        print(f"  • {cat}: {count}")

    # Filter for noteheads and accidentals
    accidental_notehead_categories = {
        'noteheadBlackInSpace', 'noteheadBlackOnLine', 'noteheadWhiteInSpace',
        'noteheadWhiteOnLine', 'noteheadHalfInSpace', 'noteheadHalfOnLine',
        'keyFlat', 'keySharp', 'keyNatural', 'accidentalFlat', 'accidentalSharp',
        'accidentalNatural', 'accidentalDoubleSharp', 'accidentalDoubleFlat'
    }

    filtered_predictions = [
        pred for pred in result.object_prediction_list
        if pred.category.name in accidental_notehead_categories
    ]

    filtered_categories = Counter([pred.category.name for pred in filtered_predictions])
    print(f"\n✓ Filtered to {len(filtered_predictions)} noteheads and accidentals:")
    for cat, count in sorted(filtered_categories.items()):
        print(f"  • {cat}: {count}")

    # Generate three output versions
    print("\nGenerating output images...")

    # 1. Everything with labels
    output1 = visualize_predictions(
        image_rgb,
        result.object_prediction_list,
        hide_labels=False,
        hide_conf=False
    )
    output1_path = 'output_1_everything.png'
    cv2.imwrite(output1_path, cv2.cvtColor(output1, cv2.COLOR_RGB2BGR))
    print(f"  ✓ {output1_path} - All objects with labels")

    # 2. Noteheads and accidentals with labels
    output2 = visualize_predictions(
        image_rgb,
        filtered_predictions,
        hide_labels=False,
        hide_conf=False
    )
    output2_path = 'output_2_filtered_with_labels.png'
    cv2.imwrite(output2_path, cv2.cvtColor(output2, cv2.COLOR_RGB2BGR))
    print(f"  ✓ {output2_path} - Noteheads & accidentals with labels")

    # 3. Noteheads and accidentals without labels
    output3 = visualize_predictions(
        image_rgb,
        filtered_predictions,
        hide_labels=True,
        hide_conf=True
    )
    output3_path = 'output_3_filtered_boxes_only.png'
    cv2.imwrite(output3_path, cv2.cvtColor(output3, cv2.COLOR_RGB2BGR))
    print(f"  ✓ {output3_path} - Noteheads & accidentals boxes only")

    print("\n✅ Done! Created 3 output images.")

if __name__ == "__main__":
    main()
