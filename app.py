from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import tempfile

app = Flask(__name__)

MODEL_PATH = "cry_detector.keras"
model = load_model(MODEL_PATH)

# Convert any audio file to WAV
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)  # supports mp3, wav, pcm, etc.
    # Normalize audio to -20 dBFS
    audio = audio.apply_gain(-20.0 - audio.dBFS)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

# Extract features for prediction
def extract_features(file_path, max_pad_len=174):
    file_path = convert_to_wav(file_path)
    audio, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
    # Trim silent edges
    audio, _ = librosa.effects.trim(audio, top_db=20)
    # Normalize amplitude
    audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

@app.route("/")
def home():
    return {"message": "Infant Cry Detection API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=(0, -1))  # (1, 40, 174, 1)
        prediction = model.predict(features)
        label_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        # Reverse logic: 1 = cry, 0 = not_cry
        label = "cry" if label_index == 0 else "not_cry"
        response = 1 if label_index == 0 else 0

        return jsonify({
            "result": response,
            "confidence": confidence,
            "label": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
