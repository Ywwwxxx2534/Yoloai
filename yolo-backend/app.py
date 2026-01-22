import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import io, base64, time, threading

app = FastAPI()
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
IMG_SIZE = 640

latest_result = {"timestamp": 0.0, "detections": []}
lock = threading.Lock()

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

def update_latest(detections):
    with lock:
        latest_result["timestamp"] = time.time()
        latest_result["detections"] = detections

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            if msg.startswith("data:image"):
                b64 = msg.split(",",1)[1]
            else:
                b64 = msg
            image_bytes = base64.b64decode(b64)
            detections = run_inference(image_bytes)
            update_latest(detections)
            await ws.send_json({"detections": detections})
    except Exception:
        await ws.close()

@app.get("/latest")
def latest():
    with lock:
        return latest_result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)