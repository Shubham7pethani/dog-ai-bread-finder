import os
import io
import logging
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VISION_MODEL = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"
KNOWLEDGE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

# Global Variables
vision_model = None
vision_processor = None
text_pipeline = None
is_loading = False
llm_is_loading = False

# ---------------- LIGHTWEIGHT LOADING ----------------

def load_essentials():
    """Loads ONLY the vision model first to save RAM at startup."""
    global vision_model, vision_processor, is_loading
    if is_loading or vision_model: return
    is_loading = True
    try:
        logger.info("👀 Loading Vision Model (The Eyes)...")
        vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
        vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL, use_fast=True)
        logger.info("✅ Vision Model Ready!")
    except Exception as e:
        logger.error(f"❌ Vision Load Failed: {e}")
    finally:
        is_loading = False

def load_llm():
    """Loads the Brain ONLY when needed or in the background."""
    global text_pipeline, llm_is_loading
    if llm_is_loading or text_pipeline: return
    llm_is_loading = True
    try:
        logger.info("🧠 Loading Tiny LLM Brain (Qwen 0.5B)...")
        text_pipeline = pipeline(
            "text-generation", 
            model=KNOWLEDGE_MODEL, 
            device="cpu", 
            model_kwargs={"dtype": torch.float32, "low_cpu_mem_usage": True}
        )
        logger.info("✅ LLM Brain Ready!")
    except Exception as e:
        logger.error(f"❌ LLM Load Failed: {e}")
    finally:
        llm_is_loading = False

@app.on_event("startup")
async def startup_event():
    # Load the eyes first so the server is 'Healthy' for Railway
    load_essentials()
    # Then start loading the brain in the background
    if os.getenv("PRELOAD_LLM", "false").lower() in {"1", "true", "yes"}:
        asyncio.create_task(asyncio.to_thread(load_llm))

# ---------------- API ENDPOINTS ----------------

@app.get("/status")
async def get_status():
    if text_pipeline and vision_model:
        return {"status": "ready"}
    return {"status": "loading"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global vision_model, vision_processor, text_pipeline
    
    # Final check to ensure models are there
    if not vision_model: load_essentials()
    if not text_pipeline: load_llm()
        
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Step 1: Breed ID
        inputs = vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = vision_model(**inputs)
        
        index = torch.argmax(outputs.logits, dim=-1)
        breed = vision_model.config.id2label[int(index)].replace("_", " ").title()

        # Step 2: Brief Facts
        prompt = f"<|im_start|>user\nTell me 3 quick facts about the {breed} dog breed.<|im_end|>\n<|im_start|>assistant\n"
        response = text_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        info_text = response[0]['generated_text'].split("assistant\n")[-1].strip()

        return {"breed": breed, "info": info_text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"status": "alive"}