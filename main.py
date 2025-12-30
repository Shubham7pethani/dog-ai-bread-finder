import os
import io
import logging
from pathlib import Path
import numpy as np
import onnxruntime as ort
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
# I added ?download=true to ensure it pulls the actual file, not the HTML page.
MODEL_URL = "https://huggingface.co/Shubham7pethani/dog-breed-onnx/resolve/main/dog_breed.onnx?download=true"

app = FastAPI(title="Dog Breed AI (ONNX)")

# ---------------- CORS Support ----------------
# This allows your local index.html to communicate with the Railway backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None

def download_model(url: str, dest: Path):
    if not dest.exists():
        logger.info(f"Downloading model from {url}...")
        
        # Get token from Railway variables to fix the 401 error
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        
        # Stream the download so we don't run out of RAM
        with requests.get(url, headers=headers, stream=True) as r:
            # If this fails with 404, check if the filename on Hugging Face is exactly 'dog_breed.onnx'
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("✅ Model download complete!")
    else:
        logger.info("Model already exists, skipping download.")

def load_model():
    global session
    if session is None:
        download_model(MODEL_URL, MODEL_PATH)
        logger.info("Loading ONNX session...")
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        logger.info("✅ Session ready!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def root():
    return {"status": "Dog AI is live", "model_loaded": session is not None}

@app.post("/analyze")
async def analyze_dog(file: UploadFile = File(...)):
    if session is None:
        return {"error": "Model not ready. Please wait a moment for the download to finish."}
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocessing (Must match how your model was trained)
        image = image.resize((224, 224))
        img = np.array(image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) 
        img = np.expand_dims(img, axis=0) 
        
        inputs = {session.get_inputs()[0].name: img}
        outputs = session.run(None, inputs)
        
        probs = outputs[0][0]
        idx = int(np.argmax(probs))
        
        return {
            "class_index": idx, 
            "confidence": float(probs[idx])
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": str(e)}