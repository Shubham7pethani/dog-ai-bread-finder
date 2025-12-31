import os
import io
import logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dog-ai")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Updated Model Config ---
# Using Dennis Jansen's ResNet50 which has a proper ONNX export
MODEL_REPO = "dennisjansen/resnet-50-dog-breeds"
MODEL_FILENAME = "model.onnx"
MODEL_PATH = "models/dog_breed_resnet50.onnx"
session = None

# Stanford Dogs 120 Labels
LABELS = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black-and-tan coonhound', 'Walker hound', 'English foxhound', 'Redbone', 'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound', 'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'Wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'Flat-coated retriever', 'Curly-coated retriever', 'Golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer', 'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old English sheepdog', 'Shetland sheepdog', 'Collie', 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'Miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'Entlebucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'Malamute', 'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle', 'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo', 'Dhole', 'African hunting dog']

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        logger.info(f"🚀 Downloading 120-Breed Model from {MODEL_REPO}...")
        try:
            # Download specific ONNX file
            path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir="models")
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            os.rename(path, MODEL_PATH)
            logger.info("✅ Model Downloaded Successfully!")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise e

@app.on_event("startup")
async def startup():
    global session
    download_model()
    # Use CPU Provider for Railway compatibility
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    logger.info("✨ AI Engine Online with 120 Breeds (ResNet50)!")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if session is None: return {"error": "Model not loaded"}
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        
        # Standard Normalization for ResNet (ImageNet)
        img_data = np.array(image).astype(np.float32) / 255.0
        img_data = (img_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)

        # Run Prediction
        input_name = session.get_inputs()[0].name
        raw_out = session.run(None, {input_name: img_data})
        
        # Softmax for confidence percentage
        scores = raw_out[0][0]
        probs = np.exp(scores - np.max(scores)) / np.exp(scores - np.max(scores)).sum()
        
        idx = int(np.argmax(probs))
        return {
            "breed": LABELS[idx].replace("_", " ").title(), 
            "confidence": float(probs[idx]) * 100
        }
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return {"error": str(e)}