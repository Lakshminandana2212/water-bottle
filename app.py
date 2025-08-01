from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app) # This allows your frontend to talk to your backend

# --- Your OpenCV Logic from waterbottle.py ---
def analyze_bottle_image(img):
    if img is None:
        return {"error": "Invalid image provided."}
    try:
        resized = cv2.resize(img, (400, 600))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"status": "Error", "message": "No contours found."}

        bottle_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(bottle_contour)
        roi = resized[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY_INV)

        heights = []
        for i in range(thresh.shape[0]):
            row = thresh[i, :]
            if cv2.countNonZero(row) > 0.5 * thresh.shape[1]:
                heights.append(i)

        if heights:
            water_level = max(heights)
            fill_ratio = (thresh.shape[0] - water_level) / thresh.shape[0]
            percentage = fill_ratio * 100
            return {
                "status": "Success",
                "fill_percentage": round(percentage, 2),
                "reasoning": f"Detected liquid level at {round(percentage, 2)}% capacity via contour analysis."
            }
        else:
            return { "status": "Success", "fill_percentage": 0, "reasoning": "Container appears to be empty." }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

# --- The API Endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
    if file:
        in_memory_file = file.read()
        np_arr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        result = analyze_bottle_image(img)
        return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)