from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

model = tf.keras.models.load_model("cry_detector.keras")

# Store latest command (0 or 1) for ESP32
latest_command = {"value": 0}

def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    max_pad_len = 174
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    pred = model.predict(mfccs)
    label = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return label, confidence

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    filepath = "temp.wav"
    f.save(filepath)
    label, confidence = predict_audio(filepath)

    # Update latest command for ESP32
    latest_command["value"] = label

    response_json = {"label": "cry" if label==1 else "not_cry", "confidence": confidence}
    return jsonify(response_json)

# ESP32 polls this endpoint to get command
@app.route("/get_command", methods=["GET"])
def get_command():
    return str(latest_command["value"])

# Web page can update command manually
@app.route("/set_command/<int:cmd>", methods=["POST"])
def set_command(cmd):
    if cmd not in [0, 1]:
        return "Invalid command", 400
    latest_command["value"] = cmd
    return f"Command set to {cmd}"

if __name__ == "__main__":
    from pyngrok import ngrok

    NGROK_AUTHTOKEN = "32nzRcM90TFzZMyd6dzoLDjRrlT_j22TT31fZiacLYiwtnZ4"
    ngrok.set_auth_token(NGROK_AUTHTOKEN)

    public_url = ngrok.connect(5000)
    print("🌐 Public URL:", public_url)
    app.run()
