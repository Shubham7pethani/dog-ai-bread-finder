import os
import io
import logging
import asyncio
from pathlib import Path
import json
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-ai")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VISION_MODEL = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"
KNOWLEDGE_MODEL = os.getenv("KNOWLEDGE_MODEL", "google/flan-t5-small")
KNOWLEDGE_TASK = os.getenv("KNOWLEDGE_TASK", "text2text-generation")

# Global Variables
vision_model = None
vision_processor = None
text_pipeline = None
is_loading = False
llm_is_loading = False
breed_info_cache = {}
wiki_cache = {}

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
        logger.info(f"🧠 Loading Tiny LLM Brain ({KNOWLEDGE_MODEL})...")
        text_pipeline = pipeline(
            KNOWLEDGE_TASK,
            model=KNOWLEDGE_MODEL, 
            device="cpu", 
            model_kwargs={"dtype": torch.float32, "low_cpu_mem_usage": True}
        )
        logger.info("✅ LLM Brain Ready!")
    except Exception as e:
        logger.error(f"❌ LLM Load Failed: {e}")
    finally:
        llm_is_loading = False

def generate_breed_info(breed: str) -> str:
    if KNOWLEDGE_TASK == "text2text-generation":
        prompt = (
            f"Write a detailed guide about the {breed} dog breed with headings. "
            "Include: Overview, Temperament, Exercise needs, Grooming, Training tips, "
            "Common health issues, Good for families?, and 3 fun facts."
        )
        response = text_pipeline(
            prompt,
            max_new_tokens=260,
            min_new_tokens=140,
            num_beams=4,
            do_sample=False,
        )
        return response[0]["generated_text"].strip()

    prompt = f"""
You are an assistant that provides information on dog breeds.
User: Tell me 3 quick facts about the {breed} dog breed.
Assistant: """
    response = text_pipeline(prompt, max_new_tokens=120, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].split("Assistant: ")[-1].strip()

def fetch_wikipedia_facts(breed: str) -> str | None:
    key = breed.strip().lower()
    if not key:
        return None
    if key in wiki_cache:
        return wiki_cache[key]

    title = quote(breed.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "dog-ai-breed-finder/1.0 (contact: none)",
                "Accept": "application/json",
            },
        )
        with urlopen(req, timeout=3) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
        extract = (data.get("extract") or "").strip()
        if not extract:
            return None

        extra = (
            "\n\nOwner quick guide:\n"
            "- Exercise: daily walks + play, mental stimulation.\n"
            "- Training: short positive sessions, reward-based.\n"
            "- Grooming: brush regularly, keep ears clean.\n"
            "- Health: regular vet checkups and vaccinations.\n"
        )
        text = extract + extra
        wiki_cache[key] = text
        return text
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None

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
    vision_ready = vision_model is not None and vision_processor is not None
    llm_ready = text_pipeline is not None
    return {
        "status": "ready" if (vision_ready and llm_ready) else ("online" if vision_ready else "loading"),
        "vision_ready": vision_ready,
        "llm_ready": llm_ready,
        "llm_loading": llm_is_loading,
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global vision_model, vision_processor, text_pipeline
    
    # Final check to ensure models are there
    if not vision_model: load_essentials()
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Step 1: Breed ID
        inputs = vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = vision_model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)
        index = torch.argmax(probs, dim=-1)
        confidence = float(probs[0, int(index)].item() * 100.0)
        breed = vision_model.config.id2label[int(index)].replace("_", " ").title()

        if not text_pipeline:
            if not llm_is_loading:
                asyncio.create_task(asyncio.to_thread(load_llm))
            return {
                "breed": breed,
                "confidence": round(confidence, 1),
                "llm_ready": False,
                "llm_loading": llm_is_loading,
                "info": None,
            }

        info_text = generate_breed_info(breed)
        return {
            "breed": breed,
            "confidence": round(confidence, 1),
            "llm_ready": True,
            "llm_loading": False,
            "info": info_text,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facts")
async def facts(breed: str):
    global text_pipeline
    if not breed:
        raise HTTPException(status_code=400, detail="MISSING_BREED")

    breed_key = breed.strip().lower()
    if breed_key in breed_info_cache:
        return {
            "status": "ready",
            "llm_ready": True,
            "source": "cache",
            "breed": breed,
            "info": breed_info_cache[breed_key],
        }

    if not text_pipeline:
        if not llm_is_loading:
            asyncio.create_task(asyncio.to_thread(load_llm))

        wiki = fetch_wikipedia_facts(breed)
        if wiki:
            return {
                "status": "ready",
                "llm_ready": False,
                "source": "wiki",
                "breed": breed,
                "info": wiki,
            }

        return JSONResponse(status_code=202, content={"status": "loading", "llm_ready": False})

    info_text = generate_breed_info(breed)
    breed_info_cache[breed_key] = info_text
    return {"status": "ready", "llm_ready": True, "source": "llm", "breed": breed, "info": info_text}

@app.get("/")
async def root():
    return FileResponse(
        str(Path(__file__).with_name("index.html")),
        headers={"Cache-Control": "no-store"},
    )

@app.get("/health")
async def health():
    return {"status": "alive"}