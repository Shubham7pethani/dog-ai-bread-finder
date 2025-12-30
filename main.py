import os
import io
import logging
from pathlib import Path
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dog-ai")

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "dog_breed.onnx"

# ---------------- App Setup ----------------
app = FastAPI(title="Dog Breed AI (ONNX)")

# ---------------- CORS Support ----------------
# This allows your local index.html to talk to Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None

# ---------------- Functions ----------------

def download_model():
    """Downloads model from your shankar777 profile."""
    if not MODEL_PATH.exists():
        logger.info("🚀 Starting model download from shankar777/dog-breed-onnx...")
        try:
            hf_hub_download(
                repo_id="shankar777/dog-breed-onnx", # Corrected username
                filename="dog_breed.onnx", 
                token=os.getenv("HF_TOKEN"),
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,
                repo_type="model"
            )
            logger.info("✅ Model download complete!")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise e
    else:
        logger.info("✅ Model already exists, skipping download.")

def load_session():
    global session
    if session is None:
        download_model()
        logger.info("🔄 Loading ONNX session into memory...")
        # Railway Free Tier uses CPU
        session = ort.InferenceSession(
            str(MODEL_PATH), 
            providers=["CPUExecutionProvider"]
        )
        logger.info("✨ AI Engine Ready!")

# ---------------- Events ----------------

@app.on_event("startup")
async def startup_event():
    load_session()

# ---------------- Routes ----------------

@app.get("/")
def root():
    return {
        "status": "Dog AI is live", 
        "model_loaded": session is not None,
        "profile": "shankar777"
    }

@app.post("/analyze")
async def analyze_dog(file: UploadFile = File(...)):
    if session is None:
        return {"error": "Model not loaded yet."}
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocessing
        image = image.resize((224, 224))
        img = np.array(image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) 
        img = np.expand_dims(img, axis=0) 
        
        # Run Inference
        inputs = {session.get_inputs()[0].name: img}
        outputs = session.run(None, inputs)
        
        probs = outputs[0][0]
        idx = int(np.argmax(probs))
        
        return {
            "class_index": idx, 
            "confidence": round(float(probs[idx]), 4)
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": "Failed to process image"}