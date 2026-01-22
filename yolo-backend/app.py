import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import io, base64

app = FastAPI()

# Model yükle
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
IMG_SIZE = 640

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2,0,1))
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def run_inference(image_bytes):
    input_tensor = preprocess(image_bytes)
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    preds = outputs[0]
    results = []
    for p in preds:
        x1,y1,x2,y2,conf,cls = p
        results.append({
            "bbox":[float(x1),float(y1),float(x2),float(y2)],
            "confidence":float(conf),
            "class_id":int(cls)
        })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections = run_inference(image_bytes)
    return {"detections": detections}

@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO AI Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #1f1c2c, #928dab);
                color: white;
                text-align: center;
                padding: 50px;
            }
            h1 {
                font-size: 3em;
                margin-bottom: 20px;
            }
            .upload-box {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                display: inline-block;
            }
            input[type=file] {
                margin: 10px;
            }
            button {
                background: #ff6f61;
                border: none;
                padding: 10px 20px;
                color: white;
                font-size: 1em;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background: #ff3b2e;
            }
            #result {
                margin-top: 20px;
                font-size: 1.2em;
            }
        </style>
    </head>
    <body>
        <h1>🚀 YOLO AI Live Demo</h1>
        <p>Upload an image and see real-time predictions!</p>
        <div class="upload-box">
            <input type="file" id="fileInput">
            <button onclick="sendFile()">Predict</button>
        </div>
        <div id="result"></div>

        <script>
            async function sendFile() {
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files.length) {
                    alert("Please select a file first!");
                    return;
                }
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                const res = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                document.getElementById("result").innerText =
                    JSON.stringify(data.detections, null, 2);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)            
