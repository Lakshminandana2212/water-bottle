import cv2
import numpy as np


image_path = r"E:\waterbottledetector\waterbottle3.jpg" 

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found.")
    exit()


resized = cv2.resize(img, (400, 600))  
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)


edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No contours found.")
    exit()


bottle_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(bottle_contour)
roi = resized[y:y+h, x:x+w]


roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)


heights = []
for i in range(thresh.shape[0]):
    row = thresh[i, :]
    if cv2.countNonZero(row) > 0.5 * thresh.shape[1]:  
        heights.append(i)

if heights:
    water_level = max(heights)
    fill_ratio = (thresh.shape[0] - water_level) / thresh.shape[0]
    percentage = fill_ratio * 100
    print(f"Estimated water fill: {percentage:.2f}%")
else:
    print("Could not detect water level.")
