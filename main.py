from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, os, json, colorsys, joblib
from tensorflow.keras.models import load_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = "artifacts"
MODEL_PATH_1 = os.path.join(ARTIFACTS_DIR, "model", "keras_model.h5")
MODEL_PATH_2 = os.path.join(ARTIFACTS_DIR, "keras_model.h5")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")

model = None
label_encoder = None
scaler = None
meta = None

def _lazy_load():
    global model, label_encoder, scaler, meta
    model_path = MODEL_PATH_1 if os.path.isfile(MODEL_PATH_1) else MODEL_PATH_2
    for p in [model_path, LABEL_ENCODER_PATH, SCALER_PATH, META_PATH]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    if model is None:
        model = load_model(model_path, compile=False)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

class EmotionRequest(BaseModel):
    color_hex: str
    shape: str
    sound: str

def hex_to_hsv(x: str):
    x = str(x).strip().lstrip("#")
    if len(x) == 3:
        x = "".join(c*2 for c in x)
    r = int(x[0:2],16)/255.0
    g = int(x[2:4],16)/255.0
    b = int(x[4:6],16)/255.0
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return [h,s,v]

@app.get("/")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict_emotion(req: EmotionRequest):
    _lazy_load()
    hsv = hex_to_hsv(req.color_hex)
    hsv_scaled = scaler.transform([hsv])[0]
    n_shape = len(meta.get("shape_categories", []))
    n_sound = len(meta.get("sound_categories", []))
    zeros_shape = np.zeros(n_shape, dtype=np.float32)
    zeros_sound = np.zeros(n_sound, dtype=np.float32)
    X = np.hstack([hsv_scaled, zeros_shape, zeros_sound]).reshape(1, -1)
    probs = model.predict(X)[0]
    top = int(np.argmax(probs))
    return {
        "prediction": label_encoder.inverse_transform([top])[0],
        "confidence": float(np.max(probs)),
        "probabilities": dict(zip(label_encoder.classes_, map(float, probs)))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
