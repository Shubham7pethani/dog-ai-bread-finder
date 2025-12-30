import os
import io
import logging
from pathlib import Path
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files

# ---------------- Setup ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dog-ai")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "dog_breed.onnx"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None

# ---------------- The Model Logic ----------------

def download_specific_model():
    """Downloads ONLY the ScottMueller model."""
    if not MODEL_PATH.exists():
        repo_id = "ScottMueller/Cat_Dog_Breeds.ONNX"
        logger.info(f"🚀 Accessing repo: {repo_id}")
        
        try:
            # First, let's see what the file is actually called in that repo
            files = list_repo_files(repo_id)
            # Find the one that ends with .onnx (case insensitive)
            onnx_filename = next((f for f in files if f.lower().endswith(".onnx")), None)
            
            if not onnx_filename:
                raise FileNotFoundError(f"Could not find any .onnx file in {repo_id}")

            logger.info(f"📂 Found target file: {onnx_filename}. Downloading...")

            # Download that exact file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=onnx_filename,
                local_dir=str(MODEL_DIR)
            )
            
            # Rename it to dog_breed.onnx for our local code
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            os.rename(downloaded_path, str(MODEL_PATH))
            logger.info("✅ Specific model downloaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error downloading that model: {e}")
            raise e

@app.on_event("startup")
async def startup_event():
    global session
    download_specific_model()
    if MODEL_PATH.exists():
        logger.info("🔄 Loading ONNX session...")
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        logger.info("✨ AI Engine Ready!")

# ---------------- Routes ----------------

@app.get("/")
def status():
    return {"status": "online", "model": "ScottMueller/Cat_Dog_Breeds.ONNX", "ready": session is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if session is None: return {"error": "Model not loaded"}
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        img = np.array(image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        res = session.run(None, {session.get_inputs()[0].name: img})
        idx = int(np.argmax(res[0][0]))
        return {"class_index": idx, "confidence": float(res[0][0][idx])}
    except Exception as e:
        return {"error": str(e)}