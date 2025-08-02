from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # This enables Cross-Origin Resource Sharing for your app

# --- Updated OpenCV Logic based on waterbottle.py ---
def analyze_bottle_image(img):
    """
    Takes a CV2 image object and returns a dictionary with the analysis.
    Uses the logic from waterbottle.py for better water detection.
    """
    if img is None:
        return {"status": "Error", "message": "Invalid image provided."}

    try:
        # Resize for consistent processing (same as waterbottle.py)
        image = cv2.resize(img, (400, 600))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to isolate water (dark areas) - key logic from waterbottle.py
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"status": "Error", "message": "No contours found in the image."}
        
        # Assume largest contour is the bottle
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the bottle region
        bottle_roi = thresh[y:y+h, x:x+w]
        
        # Count dark (black) pixels - this represents water
        total_pixels = bottle_roi.size
        water_pixels = cv2.countNonZero(bottle_roi)
        
        # Calculate fill percentage
        fill_percentage = int((water_pixels / total_pixels) * 100)
        
        # Limit to 0-100 range
        fill_percentage = max(0, min(fill_percentage, 100))
        
        # Generate reasoning based on results
        if fill_percentage > 70:
            reasoning = "Bottle appears to be mostly full with high water content detected."
        elif fill_percentage > 30:
            reasoning = "Bottle has moderate water content - partially filled."
        elif fill_percentage > 5:
            reasoning = "Low water level detected in the bottle."
        else:
            reasoning = "Bottle appears to be empty or nearly empty."
        
        return {
            "status": "Success",
            "fill_percentage": fill_percentage,
            "reasoning": f"{reasoning} Detected {water_pixels} water pixels out of {total_pixels} total pixels in bottle region."
        }
        
    except Exception as e:
        return {"status": "Error", "message": f"Analysis failed: {str(e)}"}


# --- The API Endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400

    if file:
        try:
            # Read the image file from the request into memory
            in_memory_file = file.read()
            # Convert the file data to a numpy array
            np_arr = np.frombuffer(in_memory_file, np.uint8)
            # Decode the numpy array into an OpenCV image
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Pass the image to your analysis function
            result = analyze_bottle_image(img)
            
            # Return the result as JSON
            return jsonify(result)
        
        except Exception as e:
            return jsonify({"status": "Error", "message": f"Failed to process image: {str(e)}"}), 500


# --- Health check endpoint (optional) ---
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Server is running"}), 200


# --- Run the server ---
if __name__ == "__main__":
    print("Starting Water Bottle Detector API...")
    print("Server will be available at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)