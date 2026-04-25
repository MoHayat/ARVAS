"""
ARVAS Live Web Demo — FastAPI Backend

Serves a live chat interface with real-time emotional state visualization.
The model is loaded once at startup and stays in memory for fast responses.

Usage:
    source venv/bin/activate
    cd demo/web
    python app.py

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
from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering
from sentiment_trigger import SentimentTrigger

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B for fast interactive responses
DEVICE = "cpu"  # Use CPU for stability; change to "mps" if desired
DTYPE = torch.float32
MAX_NEW_TOKENS = 120
TARGET_LAYER = "model.layers.10"

# Load calibration
CALIB_PATH = PROJECT_ROOT / "outputs" / "experiment_03" / "calibration.json"
if CALIB_PATH.exists():
    with open(CALIB_PATH) as f:
        calibration = json.load(f)
    trigger_params = calibration["trigger_parameters"]
else:
    trigger_params = {
        "decay_rate": 0.6,
        "sensitivity": 1.8,
        "alpha_scale": 1.5,
        "joy_threshold": 0.2,
        "grief_threshold": -0.2,
    }

# ------------------------------------------------------------------
# Global model state (loaded on startup)
# ------------------------------------------------------------------
model = None
tokenizer = None
joy_direction = None
grief_direction = None

# Session storage (in-memory for demo)
# In production, use Redis or a database
sessions: Dict[str, Dict] = {}

# ------------------------------------------------------------------
# Lifespan context manager (startup/shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, tokenizer, joy_direction, grief_direction
    
    print("=" * 60)
    print("ARVAS Live Demo — Loading Model...")
    print("=" * 60)
    
    # Load model
    print(f"Loading {MODEL_NAME} on {DEVICE} with {DTYPE}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Model loaded: {len(model.model.layers)} layers, {model.config.hidden_size} hidden dim")
    
    # Load direction vectors (normalized)
    norm_dir = PROJECT_ROOT / "outputs" / "directions"
    joy_path = norm_dir / "joy_direction_norm.pt"
    grief_path = norm_dir / "grief_direction_norm.pt"
    
    if not joy_path.exists():
        print("  Warning: Normalized direction vectors not found.")
        print("  Run experiments 1-3 first to generate them.")
        joy_direction = None
        grief_direction = None
    else:
        joy_direction = torch.load(joy_path, weights_only=True).to(DEVICE)
        grief_direction = torch.load(grief_path, weights_only=True).to(DEVICE)
        print(f"  Directions loaded: joy norm={joy_direction.norm():.4f}, grief norm={grief_direction.norm():.4f}")
    
    print("=" * 60)
    print("Ready! Open http://localhost:8000 in your browser.")
    print("=" * 60)
    
    yield  # Server runs here
    
    # Shutdown (cleanup if needed)
    print("Shutting down...")

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="ARVAS Live Demo", version="1.0", lifespan=lifespan)

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
    emotion_level: float
    direction: str
    alpha: float
    sentiment: float
    turn: int

class ResetRequest(BaseModel):
    session_id: str

# ------------------------------------------------------------------
# Helper: Generate with steering
# ------------------------------------------------------------------
def generate_response(history: List[Dict[str, str]], direction_vec: Optional[torch.Tensor], alpha: float) -> str:
    """Generate a response from the model with optional steering."""
    # Build chat prompt
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    
    if direction_vec is not None and alpha > 0:
        return generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            layer_names=[TARGET_LAYER],
            direction=direction_vec,
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
    """Process a chat message and return the model's response with emotional state."""
    session_id = request.session_id
    user_message = request.message.strip()
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "trigger": SentimentTrigger(**trigger_params),
            "turn": 0,
        }
    
    session = sessions[session_id]
    trigger = session["trigger"]
    session["turn"] += 1
    turn = session["turn"]
    
    # Update trigger with user message
    direction_name, alpha = trigger.update(user_message)
    state = trigger.get_state()
    emotion_level = state["emotion_level"]
    
    # Get sentiment score
    sentiment = trigger.score_message(user_message)
    
    # Select direction vector
    direction_vec = None
    if direction_name == "joy" and joy_direction is not None:
        direction_vec = joy_direction
    elif direction_name == "grief" and grief_direction is not None:
        direction_vec = grief_direction
    
    # Generate response
    history = session["history"] + [{"role": "user", "content": user_message}]
    response = generate_response(history, direction_vec, alpha)
    
    # Update session history
    session["history"].append({"role": "user", "content": user_message})
    session["history"].append({"role": "assistant", "content": response})
    
    return ChatResponse(
        response=response,
        emotion_level=emotion_level,
        direction=direction_name,
        alpha=alpha,
        sentiment=sentiment,
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
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "directions_loaded": joy_direction is not None,
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
