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

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dog-ai")

# 2. Define Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "dog_breed.onnx"

app = FastAPI(title="Dog Breed AI")

# 3. Enable CORS (Crucial for your index.html)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None

# 4. Model Download Logic
def download_model():
    """Downloads model from shankar777/dog-breed-onnx."""
    if not MODEL_PATH.exists():
        logger.info("🚀 Starting download from shankar777/dog-breed-onnx...")
        try:
            # We use the token from your Railway variables
            hf_hub_download(
                repo_id="shankar777/dog-breed-onnx",
                filename="dog_breed.onnx",
                token=os.getenv("HF_TOKEN"),
                local_dir=str(MODEL_DIR)
            )
            logger.info("✅ Download finished!")
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            raise e

def load_session():
    """Initializes the AI engine."""
    global session
    if session is None:
        download_model()
        logger.info("🔄 Loading AI Engine...")
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        logger.info("✨ AI Engine Ready!")

@app.on_event("startup")
async def startup_event():
    load_session()

# 5. Routes
@app.get("/")
def health_check():
    return {"status": "online", "model_ready": session is not None}

@app.post("/analyze")
async def analyze_dog(file: UploadFile = File(...)):
    if session is None:
        return {"error": "AI not ready yet"}
    
    try:
        # Read image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Preprocess (Standard 224x224)
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1)) # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)
        
        # Inference
        input_name = session.get_inputs()[0].name
        results = session.run(None, {input_name: img_array})
        
        # Output
        probabilities = results[0][0]
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        
        return {
            "class_index": class_idx,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": "Processing failed"}