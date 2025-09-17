from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ===============================
# STEP 0: Setup
# ===============================
app = Flask(__name__)

# Load trained model
MODEL_PATH = "cry_detector.keras"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Labels encoder
encoder = LabelEncoder()
encoder.fit(["not_cry", "cry"])

# ===============================
# STEP 1: Feature Extraction
# ===============================
def extract_features(file_path, max_pad_len=174):
    audio, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# ===============================
# STEP 2: API Endpoints
# ===============================
@app.route("/")
def home():
    return {"message": "Infant cry detection API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temp file
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Extract features
    mfccs = extract_features(filepath)
    X = mfccs[np.newaxis, ..., np.newaxis]  # Add batch & channel dimension

    # Predict
    pred_probs = model.predict(X)[0]  # e.g., [0.02, 0.98]
    class_index = np.argmax(pred_probs)
    label = encoder.inverse_transform([class_index])[0]
    confidence = float(pred_probs[class_index])

    # Convert to 0/1 for ESP32
    result = 1 if label == "cry" else 0

    return jsonify({
        "result": result,         # 0 or 1 for ESP32
        "confidence": confidence, # confidence score
        "label": label            # optional human-readable
    })

# ===============================
# STEP 3: Run Server
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
