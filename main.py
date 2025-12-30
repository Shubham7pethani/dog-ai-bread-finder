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

app = FastAPI(title="Dog Breed AI")

# ---------------- CORS Support ----------------
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
    """Downloads model from ScottMueller/Cat_Dog_Breeds.ONNX to fix 404 errors."""
    if not MODEL_PATH.exists():
        logger.info("🚀 Starting download from ScottMueller/Cat_Dog_Breeds.ONNX...")
        try:
            # We download the specific file from the working repo you found
            downloaded_file_path = hf_hub_download(
                repo_id="ScottMueller/Cat_Dog_Breeds.ONNX",
                filename="Cat_Dog_Breeds.onnx",
                token=os.getenv("HF_TOKEN"),
                local_dir=str(MODEL_DIR)
            )
            
            # Rename the downloaded file to dog_breed.onnx so the rest of the code works
            os.rename(downloaded_file_path, str(MODEL_PATH))
            logger.info("✅ Model download and rename complete!")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise e
    else:
        logger.info("✅ Model already exists, skipping download.")

def load_session():
    global session
    if session is None:
        download_model()
        logger.info("🔄 Loading AI Engine...")
        # providers=["CPUExecutionProvider"] is required for Railway's free tier
        session = ort.InferenceSession(
            str(MODEL_PATH), 
            providers=["CPUExecutionProvider"]
        )
        logger.info("✨ AI Engine Ready!")

@app.on_event("startup")
async def startup_event():
    load_session()

# ---------------- Routes ----------------

@app.get("/")
def root():
    return {
        "status": "online", 
        "model_loaded": session is not None,
        "source": "ScottMueller/Cat_Dog_Breeds.ONNX"
    }

@app.post("/analyze")
async def analyze_dog(file: UploadFile = File(...)):
    if session is None:
        return {"error": "AI not ready yet. Please wait a moment."}
    
    try:
        # Read and Open Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocessing: Resize to 224x224 and normalize
        image = image.resize((224, 224))
        img = np.array(image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW format
        img = np.expand_dims(img, axis=0) # Add batch dimension
        
        # Run Inference
        inputs = {session.get_inputs()[0].name: img}
        outputs = session.run(None, inputs)
        
        # Get results
        probs = outputs[0][0]
        idx = int(np.argmax(probs))
        
        return {
            "class_index": idx, 
            "confidence": round(float(probs[idx]), 4)
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": "Failed to process image"}