import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

MODEL_ID = "prithivMLmods/Dog-Breed-120"
ONNX_PATH = "dog_breed.onnx"

print("Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

print("Preparing dummy input...")
dummy_image = Image.new("RGB", (224, 224))
inputs = processor(images=dummy_image, return_tensors="pt")

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    (inputs["pixel_values"],),
    ONNX_PATH,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=17
)

print(f"✅ ONNX model saved as {ONNX_PATH}")
