import os, io, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"

# Global variables to hold our AI "brain"
model = None
processor = None

# Simple root path so Railway sees "Success" immediately
@app.get("/")
async def root():
    status = "Ready" if model else "Loading AI..."
    return {"status": status, "model": MODEL_ID}

# ---------------- AI LOADER ----------------
def load_ai():
    global model, processor
    if model is None:
        logger.info(f"🚀 Downloading Model... this takes ~60 seconds")
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        logger.info("✅ Model Loaded and Ready!")

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Load AI on the first request if it's not ready
    if model is None:
        load_ai()
        
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        breed_name = model.config.id2label[int(index)]
        
        return {
            "breed": breed_name.replace("_", " ").title(),
            "confidence": round(float(confidence) * 100, 2)
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}