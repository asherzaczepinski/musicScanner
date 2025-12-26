#!/usr/bin/env python3
"""Quick test of musicScanner detection"""

import cv2
import numpy as np
from scanner.detection import detect_everything, visualize_predictions

# Load test image
image_path = "test_data/test_img.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Loaded image: {image.shape}")
print("Running detection...")

# Run detection
result = detect_everything(image_rgb)

print(f"Found {len(result.object_prediction_list)} objects!")

# Visualize
output_image = visualize_predictions(
    image_rgb,
    result.object_prediction_list,
    hide_conf=False
)

# Save result
output_path = "output/test_detection_output.png"
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print(f"Saved result to: {output_path}")

# Print some detected objects
print("\nFirst 10 detected objects:")
for i, pred in enumerate(result.object_prediction_list[:10]):
    print(f"  {i+1}. {pred.category.name} (confidence: {pred.score.value:.2f})")
