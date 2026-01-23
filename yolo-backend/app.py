import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

# FastAPI backend
app = FastAPI()

# Model yükle
session = ort.InferenceSession("best.onnx")

# Model giriş/çıkış isimlerini kontrol et
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

def predict_from_camera(image):
    # Kameradan gelen kareyi numpy array'e çevir
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (640, 640))  # modeline göre boyutu ayarla
    img_input = img_resized.transpose(2,0,1).astype(np.float32)[None, ...]

    # Model çalıştır
    outputs = session.run(output_names, {input_name: img_input})

    # Örnek: sınıf ve skorları döndür
    return {"classes": str(outputs[0].tolist()), "scores": str(outputs[1].tolist())}

# Gradio arayüzü (kamera otomatik açılır)
demo = gr.Interface(
    fn=predict_from_camera,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="text",
    live=True
)

# Gradio'yu FastAPI içine göm
app.mount("/", WSGIMiddleware(demo.app))