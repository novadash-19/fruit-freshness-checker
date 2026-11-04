import keras
import os
# Teachable Machine models need the older Keras 2. This line must come BEFORE
# "import tensorflow". Requires: pip install tf-keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

# ------------------------------------------------------------------
# CONFIG - matches a Teachable Machine "Keras" export
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "keras_Model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
IMG_SIZE = (224, 224)
MAX_FILE_MB = 10
# ------------------------------------------------------------------

state = {}


class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Teachable Machine models were saved with an older Keras that stored a
    'groups' argument. Newer Keras rejects it, so we drop it when loading."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def load_labels(path: Path) -> list[str]:
    """labels.txt lines look like '0 Fresh Apple' -> 'Fresh Apple'."""
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        labels.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D},
    )
    state["labels"] = load_labels(LABELS_PATH)
    print(f"Model loaded. Classes: {state['labels']}")
    yield
    state.clear()


app = FastAPI(title="Fruit Freshness API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Same preprocessing Teachable Machine uses: center-crop to 224x224,
    then scale pixels from 0..255 to -1..1."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def parse_label(label: str) -> tuple[str, str]:
    """'Fresh Apple' -> ('apple', 'fresh'), 'Rotten Banana' -> ('banana', 'rotten')."""
    words = label.lower().split()
    status = words[0]
    fruit = " ".join(words[1:]) if len(words) > 1 else "unknown"
    return fruit, status


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Max file size is {MAX_FILE_MB} MB.")

    try:
        batch = preprocess(contents)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the image file.")

    probs = state["model"].predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    fruit, status = parse_label(state["labels"][idx])

    return {
        "fruit": fruit,
        "status": status,
        "confidence": round(float(probs[idx]), 4),
    }
