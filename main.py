import os, io, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VISION_MODEL = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"
KNOWLEDGE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

vision_model = None
vision_processor = None
text_pipeline = None
is_loading = False

@app.get("/status")
async def get_status():
    global text_pipeline, is_loading
    if text_pipeline:
        return {"status": "ready"}
    if is_loading:
        return {"status": "loading"}
    return {"status": "idle"}

def load_all_ai():
    global vision_model, vision_processor, text_pipeline, is_loading
    if is_loading: return
    is_loading = True
    try:
        if vision_model is None:
            logger.info("👀 Loading Vision...")
            vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
            vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
        
        if text_pipeline is None:
            logger.info("🧠 Loading Qwen 0.5B...")
            text_pipeline = pipeline("text-generation", model=KNOWLEDGE_MODEL, device_map="cpu", torch_dtype=torch.float32)
        logger.info("✅ All AI Engines Online!")
    finally:
        is_loading = False

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global vision_model, text_pipeline
    if vision_model is None or text_pipeline is None:
        load_all_ai()
        
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = vision_model(**inputs)
        
        probs = torch.nn.functional.softmax(vision_outputs.logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        breed = vision_model.config.id2label[int(index)].replace("_", " ").title()

        prompt = f"<|im_start|>user\nBrief overview and 3 facts about {breed} dog. Short and friendly.<|im_end|>\n<|im_start|>assistant\n"
        knowledge = text_pipeline(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)
        info_text = knowledge[0]['generated_text'].split("assistant\n")[-1].strip()

        return {"breed": breed, "confidence": round(float(confidence) * 100, 2), "info": info_text}
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"status": "alive"}