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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None

def download_specific_model():
    if not MODEL_PATH.exists():
        repo_id = "ScottMueller/Cat_Dog_Breeds.ONNX"
        try:
            files = list_repo_files(repo_id)
            onnx_filename = next((f for f in files if f.lower().endswith(".onnx")), None)
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=onnx_filename, local_dir=str(MODEL_DIR))
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            os.rename(downloaded_path, str(MODEL_PATH))
            logger.info("✅ Model Ready!")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")

@app.on_event("startup")
async def startup_event():
    global session
    download_specific_model()
    if MODEL_PATH.exists():
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        logger.info("✨ AI Engine Online!")

@app.get("/")
def status():
    return {"status": "online", "ready": session is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if session is None: return {"error": "Model not loaded"}
    try:
        content = await file.read()
        # ResNet18 expects 224x224
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        img = np.array(image).astype(np.float32) / 255.0
        # Transpose to (Channels, Height, Width)
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        input_name = session.get_inputs()[0].name
        res = session.run(None, {input_name: img})
        
        # Get the top prediction
        probs = res[0][0]
        idx = int(np.argmax(probs))
        # Some models use raw scores, we ensure it's a standard float
        confidence = float(probs[idx]) 
        
        return {"class_index": idx, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}