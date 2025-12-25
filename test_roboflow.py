#!/usr/bin/env python3
"""Test musicScanner using Roboflow API"""

from roboflow import Roboflow
import cv2

# Initialize Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")  # You can use it without an API key for public models
project = rf.workspace("catundertheleaf").project("musicscanner")
model = project.version(2).model

# Load test image
image_path = "test_data/test_img.png"

print(f"Running detection on: {image_path}")

# Run inference
result = model.predict(image_path, confidence=40, overlap=50).json()

print(f"\nFound {len(result['predictions'])} objects!")

# Print first 10 detections
print("\nFirst 10 detected objects:")
for i, pred in enumerate(result['predictions'][:10]):
    print(f"  {i+1}. {pred['class']} (confidence: {pred['confidence']:.2f})")

# Save visualized result
model.predict(image_path, confidence=40, overlap=50).save("test_roboflow_output.jpg")
print(f"\nSaved visualization to: test_roboflow_output.jpg")
