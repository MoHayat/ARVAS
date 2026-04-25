# ARVAS Live Web Demo (v2)

Real-time affective reciprocity demo with **2D valence-arousal visualization**.

## What's New in v2
- **Circumplex Model visualization**: A circular emotion wheel shows the model's real-time emotional state across 8 emotions (Joy, Excitement, Calm, Boredom, Sadness, Fear, Anger, Disgust).
- **1D VU meter toggle**: The classic vertical needle gauge is still available — switch between views with the header toggle.
- **2D steering backend**: The backend now computes steering vectors from a learned valence-arousal plane rather than a single binary joy/grief axis.
- **Keyword-based arousal detection**: In addition to VADER sentiment (valence), the trigger system detects high/low arousal keywords ("furious", "terrified", "peaceful", "dull") to place the emotional state accurately on the 2D plane.

## Quick Start

```bash
cd /Users/mohayat/projects/KH/ARVAS
source venv/bin/activate

# Default: 1.5B model for fast interactive responses
cd demo/web
python app.py

# To use 7B model (requires downloading weights first):
ARVAS_MODEL=Qwen/Qwen2.5-7B-Instruct python app.py

# To override layer, device, etc.:
ARVAS_LAYER=model.layers.20 ARVAS_DEVICE=mps python app.py
```

Then open http://localhost:8000 in your browser.

## Architecture
```
Browser (index.html + app.js)
    ↕ WebSocket / HTTP
FastAPI (app.py)
    ├── AffectiveTrigger (sentiment_trigger.py)
    │   ├── VADER → valence
    │   └── Keyword heuristic → arousal
    ├── compute_2d_direction (steering.py)
    │   └── valence_axis + arousal_axis → blended direction
    └── generate_with_steering (steering.py)
        └── baukit TraceDict hook
```

## Configuration (Environment Variables)
| Variable | Default | Description |
|---|---|---|
| `ARVAS_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model name |
| `ARVAS_DEVICE` | `mps` | torch device (`mps`, `cuda`, `cpu`) |
| `ARVAS_LAYER` | auto-detected middle | Specific layer to hook |
| `ARVAS_MAX_TOKENS` | `120` | Max generation length |

## Troubleshooting
- **"No valence/arousal axes found"**: Run emotion extraction for your model size first:
  ```bash
  python src/emotion_extraction.py --model <MODEL> --stories data/emotion_stories.json --output outputs/directions
  ```
- **Slow responses on 7B**: Expected. 7B generates ~13–40 tok/s on MPS. Use 1.5B for faster demos.
- **Model feels "stuck" neutral**: Try more emotionally charged messages. The keyword heuristic needs explicit intensity words to register high arousal.

## Files
- `app.py` — FastAPI backend with lifespan model loading
- `static/index.html` — Chat UI + 1D/2D toggle + emotion wheel canvas
- `static/app.js` — Real-time gauge animation, message handling, stats
