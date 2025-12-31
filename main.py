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

# Robust CORS for Railway/Local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None

# ---------------- Model Downloading ----------------

def download_specific_model():
    """Downloads the ScottMueller ResNet18 model."""
    if not MODEL_PATH.exists():
        repo_id = "ScottMueller/Cat_Dog_Breeds.ONNX"
        logger.info(f"🚀 Accessing repo: {repo_id}")
        
        try:
            files = list_repo_files(repo_id)
            onnx_filename = next((f for f in files if f.lower().endswith(".onnx")), None)
            
            if not onnx_filename:
                raise FileNotFoundError(f"Could not find any .onnx file in {repo_id}")

            logger.info(f"📂 Downloading: {onnx_filename}")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=onnx_filename,
                local_dir=str(MODEL_DIR)
            )
            
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            os.rename(downloaded_path, str(MODEL_PATH))
            logger.info("✅ Model downloaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            raise e

@app.on_event("startup")
async def startup_event():
    global session
    download_specific_model()
    if MODEL_PATH.exists():
        logger.info("🔄 Loading ONNX session...")
        # CPU Provider for Railway Free Tier
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        logger.info("✨ AI Engine Ready!")

# ---------------- Routes ----------------

@app.get("/")
def status():
    return {"status": "online", "model": "ScottMueller", "ready": session is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if session is None: 
        return {"error": "Model not loaded on server"}
        
    try:
        # 1. Read and Resize image to 224x224 (Standard for ResNet)
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        
        # 2. Preprocess (Normalize and Transpose)
        img = np.array(image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # From HWC to CHW
        
        # 3. FIX: Batch Size Workaround
        # This model expects a batch of 10. We create 10 slots and put our image in slot 0.
        batch = np.zeros((10, 3, 224, 224), dtype=np.float32)
        batch[0] = img 
        
        # 4. Run Inference
        input_name = session.get_inputs()[0].name
        res = session.run(None, {input_name: batch})
        
        # 5. Extract results for the first image in the batch
        probs = res[0][0] 
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        
        logger.info(f"✅ Prediction: Class {idx} with {confidence:.2f} confidence")
        return {"class_index": idx, "confidence": confidence}

    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)