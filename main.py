import os, io, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

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

# ---------------- VERIFIED REPO ----------------
# This repo is public and has the best 120-breed ViT model
MODEL_ID = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"

model = None
processor = None

# ---------------- STARTUP ----------------
@app.on_event("startup")
async def startup():
    global model, processor
    logger.info(f"🚀 Loading High-Accuracy Model: {MODEL_ID}")
    try:
        # This downloads the full high-accuracy model automatically
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        logger.info("✅ Best Dog AI is ONLINE")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if model is None: return {"error": "AI not ready yet"}
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Preprocessing
        inputs = processor(images=img, return_tensors="pt")
        
        # Run through the 120-breed AI
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Calculate Percentage
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        
        # Get the actual breed name from the 120 labels
        breed_name = model.config.id2label[int(index)]
        
        return {
            "breed": breed_name.replace("_", " ").title(),
            "confidence": round(float(confidence) * 100, 2)
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}