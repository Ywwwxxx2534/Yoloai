from flask import Flask, request, jsonify, render_template_string
import onnxruntime as ort
import numpy as np
import cv2
import base64

app = Flask(__name__)
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
CLASS_NAMES = ["OpenPalm", "CrossHit", "Punch"]  # adjust to your training labels

def preprocess(bgr):
    img = cv2.resize(bgr, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :]  # BCHW
    return img

@app.route("/")
def index():
    # HTML page with live webcam prediction
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YoloAI Live Camera Prediction</title>
        <style>
            body { font-family: Arial; background:#f0f0f0; text-align:center; padding:40px; }
            h2 { color:#333; }
            #result { margin-top:20px; font-size:18px; font-weight:bold; }
            video { border:2px solid #333; margin-top:20px; }
        </style>
    </head>
    <body>
        <h2>YoloAI Live Camera Prediction</h2>
        <video id="video" width="320" height="240" autoplay></video>
        <p id="result">Starting camera...</p>
        <canvas id="canvas" width="224" height="224" style="display:none;"></canvas>
        <script>
        async function startCamera() {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            document.getElementById("video").srcObject = stream;
        }

        async function captureAndPredict() {
            const video = document.getElementById("video");
            const canvas = document.getElementById("canvas");
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, 224, 224);
            const b64 = canvas.toDataURL("image/jpeg").split(",")[1];

            try {
                const res = await fetch("/predict_base64", {
                    method: "POST",
                    headers: {"Content-Type":"application/json"},
                    body: JSON.stringify({image_b64: b64})
                });
                const data = await res.json();
                document.getElementById("result").innerText =
                    "Class: " + data.class + " | Confidence: " + data.confidence.toFixed(2);
            } catch (err) {
                document.getElementById("result").innerText = "Error: " + err;
            }
        }

        startCamera();
        setInterval(captureAndPredict, 500); // every 0.5s
        </script>
    </body>
    </html>
    """)

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    data = request.get_json(silent=True)
    if not data or "image_b64" not in data:
        return jsonify({"error": "image_b64 missing"}), 400

    try:
        raw = base64.b64decode(data["image_b64"])
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "invalid base64"}), 400

    inp = preprocess(bgr)
    outputs = session.run(None, {"images": inp})
    probs = outputs[0][0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return jsonify({"class": CLASS_NAMES[idx], "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)