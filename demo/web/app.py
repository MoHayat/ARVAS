"""
ARVAS Live Web Demo — FastAPI Backend (v2 Multi-Emotion)

Serves a live chat interface with real-time 2D emotional state visualization.
The model is loaded once at startup and stays in memory for fast responses.

Now supports 2D valence-arousal steering via the Circumplex Model.

Configuration (env vars):
    ARVAS_MODEL       — HuggingFace model name (default: Qwen/Qwen2.5-1.5B-Instruct)
    ARVAS_DEVICE      — torch device (default: mps)
    ARVAS_LAYER       — Specific layer to hook (default: auto-detected middle layer)
    ARVAS_MAX_TOKENS  — Max generation length (default: 120)

Usage:
    source venv/bin/activate
    cd demo/web
    ARVAS_MODEL=Qwen/Qwen2.5-7B-Instruct python app.py

Then open http://localhost:8000 in your browser.
"""
import sys
import os
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

from contextlib import asynccontextmanager
from activation_utils import load_model_and_tokenizer, get_layer_names
from steering import compute_2d_direction, generate_with_steering
from sentiment_trigger import AffectiveTrigger

# ------------------------------------------------------------------
# Config (override via environment variables)
# ------------------------------------------------------------------
MODEL_NAME = os.environ.get("ARVAS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE = os.environ.get("ARVAS_DEVICE", "mps")
DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32
MAX_NEW_TOKENS = int(os.environ.get("ARVAS_MAX_TOKENS", "120"))
TARGET_LAYER = os.environ.get("ARVAS_LAYER", "")  # empty = auto-detect middle layer

# Steering calibration
ALPHA_SCALE = 3.0

# Load calibration (fallback defaults)
CALIB_PATH = PROJECT_ROOT / "outputs" / "experiment_03" / "calibration.json"
if CALIB_PATH.exists():
    with open(CALIB_PATH) as f:
        calibration = json.load(f)
    trigger_params = calibration.get("trigger_parameters", {})
else:
    trigger_params = {
        "decay_rate": 0.6,
        "sensitivity": 1.8,
        "alpha_scale": 1.5,
    }

# ------------------------------------------------------------------
# Global model state (loaded on startup)
# ------------------------------------------------------------------
model = None
tokenizer = None
valence_axis = None
arousal_axis = None
model_info = {
    "name": MODEL_NAME,
    "device": DEVICE,
    "layer": TARGET_LAYER,
    "n_layers": 0,
    "hidden_size": 0,
}

# Session storage (in-memory for demo)
sessions: Dict[str, Dict] = {}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _find_directions_dir() -> Path:
    """Pick directions directory based on model name."""
    if "7B" in MODEL_NAME:
        d = PROJECT_ROOT / "outputs" / "directions_7b"
    elif "1.5B" in MODEL_NAME:
        d = PROJECT_ROOT / "outputs" / "directions"
    else:
        d = PROJECT_ROOT / "outputs" / "directions"
    return d


# ------------------------------------------------------------------
# Lifespan context manager (startup/shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, valence_axis, arousal_axis, model_info, TARGET_LAYER

    print("=" * 60)
    print("ARVAS Live Demo v2 — Multi-Emotion Spectrum")
    print("=" * 60)

    # Load model
    print(f"Loading {MODEL_NAME} on {DEVICE} with {DTYPE}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    n_layers = len(get_layer_names(model))
    hidden_size = model.config.hidden_size
    model_info["n_layers"] = n_layers
    model_info["hidden_size"] = hidden_size
    print(f"  Model loaded: {n_layers} layers, {hidden_size} hidden dim")

    # Auto-detect target layer if not set
    if not TARGET_LAYER:
        target_idx = n_layers // 2
        TARGET_LAYER = f"model.layers.{target_idx}"
        model_info["layer"] = TARGET_LAYER
        print(f"  Auto-selected middle layer: {TARGET_LAYER}")

    # Determine directions directory
    directions_dir = _find_directions_dir()
    print(f"  Looking for directions in: {directions_dir}")

    # Load valence/arousal axes for 2D steering
    axes_path = directions_dir / f"valence_arousal_axes_{TARGET_LAYER.replace('.', '_')}.pt"
    if not axes_path.exists():
        alt = list(directions_dir.glob("valence_arousal_axes_*.pt"))
        if alt:
            axes_path = alt[0]
            print(f"  Warning: Target layer axes not found. Falling back to {axes_path.name}")
        else:
            print("  Warning: No valence/arousal axes found. 2D steering disabled.")
            print(f"    Run: python src/emotion_extraction.py --model {MODEL_NAME} --output {directions_dir}")
            valence_axis = None
            arousal_axis = None

    if axes_path.exists():
        axes = torch.load(axes_path, weights_only=True).to(DEVICE)
        valence_axis = axes[0]
        arousal_axis = axes[1]
        print(f"  Loaded 2D axes from {axes_path.name}")
        print(f"    Valence axis norm: {valence_axis.norm():.4f}")
        print(f"    Arousal axis norm: {arousal_axis.norm():.4f}")

    print("=" * 60)
    print("Ready! Open http://localhost:8000 in your browser.")
    print("=" * 60)

    yield

    print("Shutting down...")

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="ARVAS Live Demo v2", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    valence: float
    arousal: float
    alpha: float
    sentiment: float
    arousal_score: float
    turn: int

class ResetRequest(BaseModel):
    session_id: str

# ------------------------------------------------------------------
# Helper: Generate with 2D steering
# ------------------------------------------------------------------
def generate_response(
    history: List[Dict[str, str]],
    valence: float,
    arousal: float,
    alpha: float,
) -> str:
    """Generate a response from the model with optional 2D steering."""
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

    if valence_axis is not None and arousal_axis is not None and alpha > 0:
        direction = compute_2d_direction(valence_axis, arousal_axis, valence, arousal)
        return generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            layer_names=[TARGET_LAYER],
            direction=direction,
            alpha=alpha,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the model's response with 2D emotional state."""
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "trigger": AffectiveTrigger(**trigger_params),
            "turn": 0,
        }

    session = sessions[session_id]
    trigger = session["trigger"]
    session["turn"] += 1
    turn = session["turn"]

    valence_level, arousal_level, alpha = trigger.update(user_message)

    sentiment = trigger.score_valence(user_message)
    arousal_score = trigger.score_arousal(user_message)

    history = session["history"] + [{"role": "user", "content": user_message}]
    response = generate_response(history, valence_level, arousal_level, alpha)

    session["history"].append({"role": "user", "content": user_message})
    session["history"].append({"role": "assistant", "content": response})

    return ChatResponse(
        response=response,
        valence=valence_level,
        arousal=arousal_level,
        alpha=alpha,
        sentiment=sentiment,
        arousal_score=arousal_score,
        turn=turn,
    )

@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset a conversation session."""
    session_id = request.session_id
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "reset"}

@app.get("/status")
async def status():
    """Check if the model is loaded and ready."""
    return {
        "model_loaded": model is not None,
        "model_name": model_info["name"],
        "device": model_info["device"],
        "axes_loaded": valence_axis is not None,
        "target_layer": model_info["layer"],
        "n_layers": model_info["n_layers"],
        "hidden_size": model_info["hidden_size"],
    }

# ------------------------------------------------------------------
# Static files
# ------------------------------------------------------------------
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
