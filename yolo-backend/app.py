from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2

app = Flask(__name__)
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

CLASS_NAMES = ["Acik", "Capraz", "Yumruk"]  # Eğitimdeki sınıf sırası

def preprocess(bgr):
    img = cv2.resize(bgr, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :]  # BCHW
    return img

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "image file missing"}), 400

    data = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "invalid image"}), 400

    inp = preprocess(bgr)
    outputs = session.run(None, {"images": inp})
    probs = outputs[0][0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return jsonify({"class": CLASS_NAMES[idx], "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)