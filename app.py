from flask import Flask, request, jsonify, render_template, url_for
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = "static/uploads"  # Store images in 'static/uploads'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER  # Set it in app config
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

# Load Model
MODEL_PATH = "pneumonia_detector.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    print("⚠️ Model file not found. Ensure 'pneumonia_detector.h5' exists in the project directory.")

# Prediction Function
def predict_pneumonia(img_path):
    if model is None:
        return "Model not loaded"

    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    return "PNEUMONIA" if prediction > 0.5 else "NORMAL"

# Homepage Route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save File to 'static/uploads/'
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Convert full path to a relative path for rendering in HTML
    img_url = url_for("static", filename=f"uploads/{file.filename}")

    # Predict
    result = predict_pneumonia(file_path)

    # Render result in the UI
    return render_template("index.html", prediction=result, img_path=img_url)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
