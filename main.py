from __future__ import annotations
import io
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dog-ai")

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "dog_breed.onnx"

# ---------------- Model URL ----------------
# Make sure this is a REAL download link, or the app will fail!
MODEL_URL = "https://huggingface.co/Shubham7pethani/dog-breed-onnx/resolve/main/dog_breed.onnx"  # Replace with your file URL

# ---------------- App ----------------
app = FastAPI(title="Dog Breed AI (ONNX)")

# ---------------- Model Session ----------------
session: ort.InferenceSession | None = None

# ---------------- Functions ----------------
def download_model(url: str, dest: Path):
    """Download ONNX model from URL if not already present."""
    if not dest.exists():
        logger.info(f"Downloading ONNX model from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("ONNX model downloaded successfully")

def load_model():
    global session
    if session is None:
        # Ensure model exists
        download_model(MODEL_URL, MODEL_PATH)

        logger.info("Loading ONNX model...")
        session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        logger.info("ONNX model loaded successfully")

def preprocess(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for ONNX model."""
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # BCHW
    return img

# ---------------- FastAPI Events ----------------
@app.on_event("startup")
def startup_event():
    load_model()

# ---------------- Routes ----------------
@app.get("/")
def root():
    return {"status": "Dog AI backend running (ONNX)"}

@app.post("/analyze")
async def analyze_dog(file: UploadFile = File(...)):
    if session is None:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess(image)
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)

    probs = outputs[0][0]
    idx = int(np.argmax(probs))

    return {
        "class_index": idx,
        "confidence": float(probs[idx])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
