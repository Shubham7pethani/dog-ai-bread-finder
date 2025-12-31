import os, io, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model IDs
VISION_MODEL = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"
KNOWLEDGE_MODEL = "microsoft/Phi-3-mini-4k-instruct" # The smart talking brain

# Global State
vision_model = None
vision_processor = None
text_pipeline = None

# ---------------- STARTUP ----------------
@app.get("/")
async def root():
    return {"status": "AI Hub Online", "vision": "Ready" if vision_model else "Loading"}

def load_all_ai():
    global vision_model, vision_processor, text_pipeline
    
    # 1. Load the "Eyes" (Vision)
    if vision_model is None:
        logger.info("👀 Loading Vision Model...")
        vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
        vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
    
    # 2. Load the "Brain" (Text Generation)
    if text_pipeline is None:
        logger.info("🧠 Loading Knowledge Model (Phi-3)...")
        # We use device_map="auto" to handle memory efficiently
        text_pipeline = pipeline(
            "text-generation",
            model=KNOWLEDGE_MODEL,
            model_kwargs={"torch_dtype": torch.float32, "trust_remote_code": True}
        )
    logger.info("✅ All AI Engines Online!")

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if vision_model is None or text_pipeline is None:
        load_all_ai()
        
    try:
        # STEP 1: Identify the Breed
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        inputs = vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = vision_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        breed = vision_model.config.id2label[int(index)].replace("_", " ").title()

        # STEP 2: Ask the "Brain" about this breed
        prompt = f"<|user|>\nTell me 3 interesting facts about the {breed} dog breed. Keep it short and friendly.<|end|>\n<|assistant|>\n"
        
        knowledge = text_pipeline(prompt, max_new_tokens=150, temperature=0.7)
        info_text = knowledge[0]['generated_text'].split("<|assistant|>\n")[-1]

        return {
            "breed": breed,
            "confidence": round(float(confidence) * 100, 2),
            "info": info_text
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}