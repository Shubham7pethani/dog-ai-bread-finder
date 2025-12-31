import os, io, json, logging
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

# ---------------- MODEL CONFIG (VERIFIED) ----------------
MODEL_REPO = "ScottMueller/Cat_Dog_Breeds.ONNX"
MODEL_FILE = "model.onnx"

# 37 breed labels (Oxford-IIIT Pets)
LABELS = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "Bengal", "Birman", "Bombay",
    "boxer", "British_Shorthair", "chihuahua", "Egyptian_Mau",
    "english_cocker_spaniel", "english_setter", "german_shorthaired",
    "great_pyrenees", "havanese", "japanese_chin", "keeshond",
    "leonberger", "Maine_Coon", "miniature_pinscher",
    "newfoundland", "Persian", "pomeranian", "pug",
    "Ragdoll", "Russian_Blue", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "Siamese", "Sphynx",
    "staffordshire_bull_terrier", "wheaten_terrier",
    "yorkshire_terrier"
]

session = None

# ---------------- HELPERS ----------------
def download_model():
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        logger.info("⬇ Downloading Pet Breed ONNX model...")
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        os.rename(downloaded, model_path)
    return model_path


def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # CHW
    return np.expand_dims(img, axis=0)


# ---------------- STARTUP ----------------
@app.on_event("startup")
async def startup():
    global session
    logger.info("🚀 Starting Pet AI Engine")

    model_path = download_model()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    logger.info("✅ Model loaded successfully")
    logger.info("🐶🐱 Pet Breed AI READY")


# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        x = preprocess(img)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: x})[0][0]

        idx = int(np.argmax(outputs))
        confidence = float(outputs[idx])

        return {
            "breed": LABELS[idx],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        logger.error(str(e))
        return {"error": str(e)}
