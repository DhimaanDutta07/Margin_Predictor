from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import os

from .routes import router  # relative import since you have __init__.py

APP_TITLE = "Delivery Margin Predictor"
MODEL_PATH = "artifacts/model/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

pipe = joblib.load(MODEL_PATH)

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# serve frontend
@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
