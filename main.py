import os, io, logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
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

# Global State
vision_model = None
vision_processor = None
text_pipeline = None
is_loading = False

# ---------------- STARTUP ----------------
@app.get("/")
async def root():
    # This responds instantly so Railway knows the app is alive!
    status = "Ready" if text_pipeline else ("Downloading Models..." if is_loading else "Waiting for first request")
    return {
        "status": status, 
        "vision": VISION_MODEL, 
        "knowledge": KNOWLEDGE_MODEL,
        "note": "First analysis will take 3-5 mins to download the brain."
    }

def load_all_ai():
    global vision_model, vision_processor, text_pipeline, is_loading
    if is_loading: return
    
    is_loading = True
    try:
        # 1. Load the "Eyes"
        if vision_model is None:
            logger.info("👀 Loading Vision Model...")
            vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
            vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL, use_fast=True)
        
        # 2. Load the "Brain" (Knowledge Model)
        if text_pipeline is None:
            logger.info("🧠 Loading Knowledge Model (Phi-3)... This takes 3-5 mins.")
            text_pipeline = pipeline(
                "text-generation",
                model=KNOWLEDGE_MODEL,
                trust_remote_code=True,
                device_map="auto",
                model_kwargs={"dtype": torch.float32} # Fixed: dtype instead of torch_dtype
            )
        logger.info("✅ All AI Engines Online!")
    except Exception as e:
        logger.error(f"❌ Load Error: {e}")
    finally:
        is_loading = False

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global vision_model, text_pipeline
    
    # If models aren't ready, start loading and tell user to wait
    if vision_model is None or text_pipeline is None:
        load_all_ai()
        return {"error": "AI is still warming up! Please try again in 2 minutes while we download the brain."}
        
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # STEP 1: Identify the Breed
        inputs = vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = vision_model(**inputs)
        
        probs = torch.nn.functional.softmax(vision_outputs.logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        breed = vision_model.config.id2label[int(index)].replace("_", " ").title()

        # STEP 2: Ask the "Brain" (Phi-3)
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
        logger.error(f"❌ Error during analysis: {e}")
        return {"error": str(e)}