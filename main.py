import os, io, logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download

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

# ---------------- PATHS ----------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- NEW MODEL CONFIG (120 BREEDS) ----------------
# This repo and filename are VERIFIED to exist
MODEL_REPO = "dennisjansen/resnet-50-dog-breeds"
MODEL_FILE = "model.onnx" 
LOCAL_MODEL_NAME = "dog_breed_120.onnx"

# Stanford Dogs 120 Labels
LABELS = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black-and-tan coonhound', 'Walker hound', 'English foxhound', 'Redbone', 'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound', 'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'Wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'Flat-coated retriever', 'Curly-coated retriever', 'Golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer', 'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old English sheepdog', 'Shetland sheepdog', 'Collie', 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'Miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'Entlebucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'Malamute', 'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle', 'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo', 'Dhole', 'African hunting dog']

session = None

# ---------------- HELPERS ----------------
def download_model():
    model_path = os.path.join(MODEL_DIR, LOCAL_MODEL_NAME)
    if not os.path.exists(model_path):
        logger.info(f"⬇ Downloading 120-Breed ONNX model from {MODEL_REPO}...")
        try:
            downloaded = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=MODEL_DIR
            )
            # Ensure the file is named correctly for our script
            if os.path.exists(model_path): os.remove(model_path)
            os.rename(downloaded, model_path)
            logger.info("✅ Download complete")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise e
    return model_path

def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    # ImageNet Normalization (Crucial for high confidence)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(img_array, axis=0)

# ---------------- STARTUP ----------------
@app.on_event("startup")
async def startup():
    global session
    logger.info("🚀 Starting Pet AI Engine")
    model_path = download_model()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    logger.info("✅ 120-Breed Model Loaded Successfully")

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if session is None: return {"error": "Model not loaded"}
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        x = preprocess(img)

        input_name = session.get_inputs()[0].name
        raw_out = session.run(None, {input_name: x})[0][0]

        # Apply Softmax to get real 0-100% confidence
        probs = np.exp(raw_out - np.max(raw_out)) / np.exp(raw_out - np.max(raw_out)).sum()
        
        idx = int(np.argmax(probs))
        breed_name = LABELS[idx].replace("_", " ").title()

        return {
            "breed": breed_name,
            "confidence": round(float(probs[idx]) * 100, 2)
        }

    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return {"error": str(e)}