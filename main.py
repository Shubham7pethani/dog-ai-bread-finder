import os
import io
import json
import logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

# ---------------- APP ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL CONFIG ----------------
MODEL_REPO = "onnx/models"
MODEL_SUBDIR = "vision/classification/efficientnet-lite4"
MODEL_FILE = "efficientnet-lite4-11.onnx"

MODEL_PATH = f"models/{MODEL_FILE}"
LABELS_PATH = "models/imagenet_labels.json"

session = None
labels = []

# ---------------- DOWNLOAD MODEL ----------------
def download_model():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        logger.info("🚀 Downloading EfficientNet-Lite4 ONNX model...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=f"{MODEL_SUBDIR}/{MODEL_FILE}",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        os.rename(model_path, MODEL_PATH)
        logger.info("✅ Model downloaded")

    if not os.path.exists(LABELS_PATH):
        logger.info("📥 Downloading ImageNet labels...")
        labels_path = hf_hub_download(
            repo_id="huggingface/label-files",
            filename="imagenet-1k.json",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        os.rename(labels_path, LABELS_PATH)
        logger.info("✅ Labels downloaded")

# ---------------- STARTUP ----------------
@app.on_event("startup")
async def startup():
    global session, labels

    download_model()

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    logger.info("🐶🐱 Pet AI Engine is ONLINE")

# ---------------- IMAGE UTILS ----------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if session is None:
        return {"error": "Model not loaded"}

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        input_tensor = preprocess_image(image)

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor})[0][0]

        probs = np.exp(output - np.max(output))
        probs /= probs.sum()

        idx = int(np.argmax(probs))
        label = labels[str(idx)]

        return {
            "prediction": label,
            "confidence": round(float(probs[idx]) * 100, 2)
        }

    except Exception as e:
        logger.error(e)
        return {"error": str(e)}
