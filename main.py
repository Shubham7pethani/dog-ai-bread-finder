import os
import io
import logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dog-ai")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "models/dog_breed_120.onnx"
session = None

# 120 Breeds List (Stanford Dogs Order)
LABELS = [
    "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", "Papillon", "Toy terrier", "Rhodesian ridgeback", "Afghan hound", "Basset", "Beagle", "Bloodhound", "Bluetick", "Black-and-tan coonhound", "Walker hound", "English foxhound", "Redbone", "Borzoi", "Irish wolfhound", "Italian greyhound", "Whippet", "Ibizan hound", "Norwegian elkhound", "Otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier", "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "Wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier", "Airedale", "Cairn", "Australian terrier", "Dandie Dinmont", "Boston bull", "Miniature schnauzer", "Giant schnauzer", "Standard schnauzer", "Scotch terrier", "Tibetan terrier", "Silky terrier", "Soft-coated wheaten terrier", "West Highland white terrier", "Lhasa", "Flat-coated retriever", "Curly-coated retriever", "Golden retriever", "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer", "Vizsla", "English setter", "Irish setter", "Gordon setter", "Brittany spaniel", "Clumber", "English springer", "Welsh springer spaniel", "Cocker spaniel", "Sussex spaniel", "Irish water spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard", "Kelpie", "Komondor", "Old English sheepdog", "Shetland sheepdog", "Collie", "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd", "Doberman", "Miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller", "Entlebucher", "Boxer", "Bull mastiff", "Tibetan mastiff", "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "Malamute", "Siberian husky", "Affenpinscher", "Basenji", "Pug", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian", "Chow", "Keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "Toy poodle", "Miniature poodle", "Standard poodle", "Mexican hairless", "Dingo", "Dhole", "African hunting dog"
]

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        logger.info("🚀 Downloading 120-Breed Model...")
        path = hf_hub_download(repo_id="prithivMLmods/Dog-Breed-120", filename="model.onnx", local_dir="models")
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        os.rename(path, MODEL_PATH)

@app.on_event("startup")
async def startup():
    global session
    download_model()
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    logger.info("✨ AI Engine Online with 120 Breeds!")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        img_data = np.array(image).astype(np.float32) / 255.0
        # ImageNet Normalization
        img_data = (img_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)

        raw_out = session.run(None, {session.get_inputs()[0].name: img_data})
        scores = raw_out[0][0]
        probs = np.exp(scores - np.max(scores)) / np.exp(scores - np.max(scores)).sum()
        
        idx = int(np.argmax(probs))
        return {
            "breed": LABELS[idx], 
            "confidence": float(probs[idx]) * 100
        }
    except Exception as e:
        return {"error": str(e)}