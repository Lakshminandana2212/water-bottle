# waterbottle4.py
import cv2
import numpy as np

def estimate_water_level(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    # Resize for consistent processing
    image = cv2.resize(image, (400, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate water (dark areas)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No contours found!"

    # Assume largest contour is the bottle
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the bottle region
    bottle_roi = thresh[y:y+h, x:x+w]

    # Count dark (black) pixels
    total_pixels = bottle_roi.size
    water_pixels = cv2.countNonZero(bottle_roi)
    fill_percentage = int((water_pixels / total_pixels) * 100)

    # Limit to 0-100
    fill_percentage = max(0, min(fill_percentage, 100))

    return f"Estimated Water Level: {fill_percentage}%"