import os
import io
import logging
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch

# ---------------- CONFIG ----------------
# Setting up logs so you can see the download progress in Railway
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

app = FastAPI()

# Allow your index.html to talk to this server
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Model IDs - Using the stable 0.5B model to fit in Railway RAM
VISION_MODEL = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"
KNOWLEDGE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

# Global Variables
vision_model = None
vision_processor = None
text_pipeline = None
is_loading = False

# ---------------- AI LOADING LOGIC ----------------

def load_all_ai():
    """This function downloads the models and loads them into memory."""
    global vision_model, vision_processor, text_pipeline, is_loading
    
    if is_loading or text_pipeline:
        return
        
    is_loading = True
    try:
        logger.info("🚀 Starting AI Engine Download (This takes 2-3 mins)...")
        
        # 1. Load the Vision Model (The Eyes)
        if vision_model is None:
            logger.info("👀 Downloading Vision Model...")
            vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
            vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
        
        # 2. Load the Tiny LLM (The Brain)
        if text_pipeline is None:
            logger.info("🧠 Downloading Qwen 0.5B (Tiny but smart)...")
            text_pipeline = pipeline(
                "text-generation", 
                model=KNOWLEDGE_MODEL, 
                device_map="cpu", 
                torch_dtype=torch.float32
            )
            
        logger.info("✅ All AI Engines Online! Ready for requests.")
    except Exception as e:
        logger.error(f"❌ Critical Error during download: {e}")
    finally:
        is_loading = False

# THIS IS THE FIX: It triggers the download as soon as the server starts!
@app.on_event("startup")
async def startup_event():
    # We run this in a separate thread so it doesn't block the server from starting
    asyncio.create_task(asyncio.to_thread(load_all_ai))

# ---------------- API ENDPOINTS ----------------

@app.get("/status")
async def get_status():
    """Used by index.html to check if the AI is ready or still downloading."""
    if text_pipeline:
        return {"status": "ready"}
    if is_loading:
        return {"status": "loading"}
    return {"status": "idle"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Main route to identify the dog and generate facts."""
    global vision_model, vision_processor, text_pipeline
    
    # Safety check: If for some reason it's not loaded, try loading it
    if text_pipeline is None:
        load_all_ai()
        return {"error": "AI is still warming up. Give it 1 more minute!"}
        
    try:
        # Read the uploaded image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Step 1: Identify the breed
        inputs = vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = vision_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        breed = vision_model.config.id2label[int(index)].replace("_", " ").title()

        # Step 2: Use the LLM to generate facts
        logger.info(f"🧠 Generating facts for: {breed}")
        prompt = f"<|im_start|>user\nProvide a brief overview and 3 amazing facts about the {breed} dog breed. Keep it short and friendly.<|im_im_end|>\n<|im_start|>assistant\n"
        
        with torch.no_grad():
            response = text_pipeline(
                prompt, 
                max_new_tokens=150, 
                temperature=0.7, 
                do_sample=True
            )
        
        # Clean up the text response
        raw_text = response[0]['generated_text']
        info_text = raw_text.split("assistant\n")[-1].strip()

        return {
            "breed": breed,
            "confidence": round(float(confidence) * 100, 2),
            "info": info_text
        }
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Server is running. AI is downloading in the background."}