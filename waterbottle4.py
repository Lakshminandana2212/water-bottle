import cv2
import numpy as np

def estimate_water_level(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    # Resize for consistent processing (ideal for bottle dimensions)
    image = cv2.resize(image, (400, 600))  # 2:3 ratio
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue color range for water in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create mask where blue colors are white and the rest are black
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No contours found!"

    # Find the largest contour (assumed to be water)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Estimate fill percentage using bounding box of water
    water_top = y
    water_bottom = y + h
    bottle_height = image.shape[0]

    # Convert height to percentage (invert because y=0 is top)
    fill_percentage = int(((bottle_height - water_top) / bottle_height) * 100)
    fill_percentage = max(0, min(fill_percentage, 100))  # Clamp between 0-100

    return f"Estimated Water Level: {fill_percentage}%"