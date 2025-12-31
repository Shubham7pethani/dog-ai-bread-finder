import os, io, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Model IDs
VISION_MODEL = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"
KNOWLEDGE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Global State (Start as None)
vision_model = None
vision_processor = None
text_pipeline = None

# ---------------- STARTUP ----------------
@app.get("/")
async def root():
    # This responds INSTANTLY so Railway knows the app is alive
    status = "Ready" if text_pipeline else "AI is warming up (Downloading Brain...)"
    return {"status": status, "note": "First analysis will trigger the 2GB download."}

def load_all_ai():
    global vision_model, vision_processor, text_pipeline
    
    # Only load if they don't exist yet
    if vision_model is None:
        logger.info("👀 Loading Vision Model...")
        vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
        vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL, use_fast=True)
    
    if text_pipeline is None:
        logger.info("🧠 Loading Knowledge Model (Phi-3)...")
        text_pipeline = pipeline(
            "text-generation",
            model=KNOWLEDGE_MODEL,
            trust_remote_code=True,
            device_map="auto",
            model_kwargs={"dtype": torch.float32} # Fixed deprecation warning
        )
    logger.info("✅ All AI Engines Online!")

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Trigger load only when someone actually uses the app
    if vision_model is None or text_pipeline is None:
        logger.info("🚀 First request detected! Starting heavy download...")
        load_all_ai()
        
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Identify Breed
        inputs = vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = vision_model(**inputs)
        
        probs = torch.nn.functional.softmax(vision_outputs.logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        breed = vision_model.config.id2label[int(index)].replace("_", " ").title()

        # Generate Facts
        prompt = f"<|user|>\nTell me 3 amazing facts about the {breed} dog breed. Keep it friendly and short.<|end|>\n<|assistant|>\n"
        
        with torch.no_grad():
            knowledge = text_pipeline(
                prompt, 
                max_new_tokens=150, 
                temperature=0.7,
                do_sample=True,
                clean_up_tokenization_spaces=True
            )
        
        raw_text = knowledge[0]['generated_text']
        info_text = raw_text.split("<|assistant|>\n")[-1].strip() if "<|assistant|>\n" in raw_text else raw_text.strip()

        return {
            "breed": breed,
            "confidence": round(float(confidence) * 100, 2),
            "info": info_text
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {"error": str(e)}