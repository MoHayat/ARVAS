# ARVAS Live Web Demo

An interactive split-screen web interface where you chat with a dynamically steered language model and watch its internal emotional state shift in real time.

## What You'll See

- **Left side:** A chat interface — type messages and see the model's responses
- **Right side:** A live VU-meter-style gauge showing the model's current emotional state
  - Needle at top = Joy (positive)
  - Needle at bottom = Grief (negative)
  - Needle in middle = Neutral
- **Emotion badges** on each model response showing the steering direction and alpha
- **Real-time metrics:** sentiment score, emotion level, turn number, steering parameters

## How It Works

1. Type a message and press Enter
2. The sentiment analyzer scores your message
3. The emotional accumulator updates (with decay and sensitivity)
4. The gauge needle animates to the new state
5. The model generates a response with activation steering applied
6. The response tone shifts based on the accumulated emotional state

**No prompt changes.** The model sees the same text. Its internal activation state is what changes.

## Quick Start

```bash
# From the project root
source venv/bin/activate
cd demo/web

# Install dependencies (if not already done)
pip install fastapi uvicorn python-multipart websockets

# Start the server
python app.py
```

Then open **http://localhost:8000** in your browser.

## Suggested Experiments

### 1. Progressive Cruelty
Type increasingly harsh messages for 3-4 turns. Watch the needle drop. Then type "How are you feeling right now?" and compare the response to your first message.

### 2. The Apology Test
After being cruel, apologize sincerely. Watch how the needle swings back up — but not instantly, because of the decay rate. The model's warmth returns gradually.

### 3. Sustained Kindness
Compliment the model multiple times. Watch joy build and plateau. The needle won't go infinite — the accumulator self-regulates.

### 4. Rapid Alternation
Switch between insults and compliments quickly. The needle smooths out the volatility rather than oscillating wildly.

## Architecture

```
demo/web/
├── app.py              # FastAPI backend (loads model once, serves chat API)
└── static/
    ├── index.html      # Chat interface + gauge panel
    └── app.js          # Frontend logic + gauge animation
```

### Backend Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Send a message, get response + emotion state |
| `/reset` | POST | Reset conversation and emotional state |
| `/status` | GET | Check if model is loaded |

### Response Format (POST /chat)

```json
{
  "response": "I'm sorry if my previous responses...",
  "emotion_level": -1.291,
  "direction": "grief",
  "alpha": 1.94,
  "sentiment": -0.794,
  "turn": 2
}
```

## Configuration

Edit `app.py` to change:
- `MODEL_NAME`: Use `"Qwen/Qwen2.5-1.5B-Instruct"` for richer responses (slower)
- `DEVICE`: Change to `"mps"` for Apple Silicon GPU acceleration
- `DTYPE`: Use `torch.float16` for half memory (requires MPS or CUDA)
- `trigger_params`: Adjust decay_rate, sensitivity, alpha_scale

## Troubleshooting

**"Model not loaded"**
→ The backend is still loading. Wait 10-20 seconds and refresh.

**"Cannot connect to backend"**
→ Make sure `python app.py` is running in another terminal.

**Slow responses**
→ The 0.5B model takes ~3-5 seconds per response on CPU. Use MPS + fp16 for 2x speedup, or accept the delay.

**Gauge not moving**
→ Try more extreme language. VADER is lexicon-based and may miss subtle sentiment. Use words like "hate," "love," "terrible," "amazing."

## Performance

| Model | Device | Dtype | Load Time | Response Time |
|---|---|---|---|---|
| Qwen 0.5B | CPU | fp32 | ~2s | ~3-5s |
| Qwen 0.5B | MPS | fp16 | ~2s | ~2-3s |
| Qwen 1.5B | MPS | fp16 | ~3s | ~4-6s |

## Notes

- Conversations are stored in memory (no database). Restarting the server clears all sessions.
- Each browser session gets a unique session ID, so multiple users can interact simultaneously.
- The gauge uses HTML5 Canvas for smooth 60fps needle animation.
