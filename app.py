from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Infant cry detection API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temp file
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # >>> Run your model prediction here
    # e.g., result = model_predict(filepath)
    # For example purposes:
    result = {"confidence": 0.98, "label": "cry"}  

    # Convert to 0/1 for ESP32
    response = 1 if result["label"] == "cry" else 0

    return jsonify({"result": response, "confidence": result["confidence"]})


if __name__ == "__main__":
    # Render sets PORT automatically → get it from environment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
