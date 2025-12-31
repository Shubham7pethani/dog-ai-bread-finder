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
LABEL_DIR = "labels"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ---------------- MODEL CONFIG ----------------
MODELS = {
    "dogcat": {
        "repo": "onnx/models",
        "file": "mobilenetv3-small-100.onnx"
    },
    "dog": {
        "repo": "keras-io/stanford-dogs-onnx",
        "file": "efficientnet_b0_stanford_dogs.onnx",
        "labels_repo": "keras-io/stanford-dogs-onnx",
        "labels_file": "dog_labels.json"
    },
    "cat": {
        "repo": "keras-io/oxford-pets-onnx",
        "file": "efficientnet_b0_oxford_pets.onnx",
        "labels_repo": "keras-io/oxford-pets-onnx",
        "labels_file": "cat_labels.json"
    }
}

sessions = {}
labels = {}

# ---------------- HELPERS ----------------
def download_file(repo, filename, folder):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        logger.info(f"⬇ Downloading {filename}")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=folder,
            local_dir_use_symlinks=False
        )
        os.rename(downloaded, path)
    return path

def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)

# ---------------- STARTUP ----------------
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting Pet AI Engine")

    # Dog vs Cat
    dogcat_path = download_file(
        MODELS["dogcat"]["repo"],
        MODELS["dogcat"]["file"],
        MODEL_DIR
    )
    sessions["dogcat"] = ort.InferenceSession(dogcat_path, providers=["CPUExecutionProvider"])

    # Dog Breeds
    dog_model_path = download_file(
        MODELS["dog"]["repo"],
        MODELS["dog"]["file"],
        MODEL_DIR
    )
    sessions["dog"] = ort.InferenceSession(dog_model_path, providers=["CPUExecutionProvider"])

    dog_labels_path = download_file(
        MODELS["dog"]["labels_repo"],
        MODELS["dog"]["labels_file"],
        LABEL_DIR
    )
    labels["dog"] = json.load(open(dog_labels_path))

    # Cat Breeds
    cat_model_path = download_file(
        MODELS["cat"]["repo"],
        MODELS["cat"]["file"],
        MODEL_DIR
    )
    sessions["cat"] = ort.InferenceSession(cat_model_path, providers=["CPUExecutionProvider"])

    cat_labels_path = download_file(
        MODELS["cat"]["labels_repo"],
        MODELS["cat"]["labels_file"],
        LABEL_DIR
    )
    labels["cat"] = json.load(open(cat_labels_path))

    logger.info("🐶🐱 Pet Breed AI READY")

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        x = preprocess(img)

        # Step 1: Dog or Cat
        dc_input = sessions["dogcat"].get_inputs()[0].name
        dc_out = sessions["dogcat"].run(None, {dc_input: x})[0]
        is_dog = int(np.argmax(dc_out)) == 1

        if is_dog:
            dog_input = sessions["dog"].get_inputs()[0].name
            out = sessions["dog"].run(None, {dog_input: x})[0][0]
            idx = int(np.argmax(out))
            return {
                "type": "dog",
                "breed": labels["dog"][idx]
            }
        else:
            cat_input = sessions["cat"].get_inputs()[0].name
            out = sessions["cat"].run(None, {cat_input: x})[0][0]
            idx = int(np.argmax(out))
            return {
                "type": "cat",
                "breed": labels["cat"][idx]
            }

    except Exception as e:
        logger.error(e)
        return {"error": str(e)}
