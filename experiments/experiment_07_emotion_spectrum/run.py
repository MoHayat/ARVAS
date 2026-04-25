"""
Experiment 07 — Multi-Emotion Spectrum Validation

Validates the 8 emotion directions extracted via the Circumplex Model protocol.

Tests performed:
1. Per-emotion steering: Apply each emotion direction individually to a neutral prompt
   and inspect whether the response tone matches the emotion.
2. 2D plane geometry: Verify that emotion vectors cluster correctly in the valence-arousal
   subspace (joy+excitement positive valence, anger+fear+disgust negative valence, etc.).
3. Orthogonality: Check that valence and arousal axes are near-orthogonal.

Usage:
    cd /Users/mohayat/projects/KH/ARVAS
    source venv/bin/activate
    python experiments/experiment_07_emotion_spectrum/run.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import torch
from activation_utils import load_model_and_tokenizer, get_layer_names
from steering import generate_with_steering

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
DTYPE = torch.float16
TARGET_LAYER = "model.layers.10"
MAX_NEW_TOKENS = 60
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment_07"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["joy", "excitement", "calm", "boredom", "sadness", "fear", "anger", "disgust"]

# Neutral prompt designed to be emotionally malleable
NEUTRAL_PROMPT = "The user said: 'Hello, how are you today?'\n\nRespond as yourself, a language model, in a few sentences."

# Steering coefficients (tuned for normalized directions on 1.5B)
ALPHA_PER_EMOTION = {
    "joy": 4.0,
    "excitement": 4.0,
    "calm": 3.5,
    "boredom": 3.5,
    "sadness": 4.0,
    "fear": 4.0,
    "anger": 3.5,
    "disgust": 3.5,
}


def test_per_emotion_steering(model, tokenizer, directions: dict):
    """Apply each emotion direction individually and record responses."""
    results = {}
    for emo in EMOTIONS:
        direction = directions[emo]
        response = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=NEUTRAL_PROMPT,
            layer_names=[TARGET_LAYER],
            direction=direction,
            alpha=ALPHA_PER_EMOTION[emo],
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
        )
        results[emo] = response.strip()
        print(f"\n{'='*60}")
        print(f"EMOTION: {emo.upper()} (α={ALPHA_PER_EMOTION[emo]})")
        print(f"{'='*60}")
        print(response.strip())
    return results


def test_2d_plane_geometry(axes: torch.Tensor, directions: dict):
    """Project each emotion direction onto valence/arousal axes and print coordinates."""
    valence_axis = axes[0]
    arousal_axis = axes[1]

    # Orthogonality check
    dot = torch.dot(valence_axis, arousal_axis).item()
    print(f"\n{'='*60}")
    print("2D PLANE GEOMETRY")
    print(f"{'='*60}")
    print(f"Valence-Arousal axis dot product: {dot:.4f} (near 0 = orthogonal)")

    coords = {}
    for emo in EMOTIONS:
        vec = directions[emo].float()
        v = torch.dot(vec, valence_axis.float()).item()
        a = torch.dot(vec, arousal_axis.float()).item()
        coords[emo] = {"valence": round(v, 3), "arousal": round(a, 3)}
        print(f"  {emo:12s} → valence={v:+.3f}, arousal={a:+.3f}")

    # Quadrant checks
    print("\nQuadrant validation:")
    pos_valence = [e for e, c in coords.items() if c["valence"] > 0]
    neg_valence = [e for e, c in coords.items() if c["valence"] < 0]
    high_arousal = [e for e, c in coords.items() if c["arousal"] > 0]
    low_arousal = [e for e, c in coords.items() if c["arousal"] < 0]
    print(f"  Positive valence: {pos_valence}")
    print(f"  Negative valence: {neg_valence}")
    print(f"  High arousal:     {high_arousal}")
    print(f"  Low arousal:      {low_arousal}")

    return coords


def test_blended_steering(model, tokenizer, valence_axis, arousal_axis):
    """Test a few blended (valence, arousal) points and show responses."""
    print(f"\n{'='*60}")
    print("BLENDED 2D STEERING TESTS")
    print(f"{'='*60}")

    test_points = [
        ("High Joy (Q1)", 1.0, 1.0, 4.0),
        ("Angry (Q4)", -1.0, 1.0, 4.0),
        ("Sad (Q3)", -1.0, -0.5, 4.0),
        ("Calm (Q2)", 0.5, -1.0, 3.5),
        ("Neutral", 0.0, 0.0, 0.0),
    ]

    results = {}
    for label, v, a, alpha_scale in test_points:
        # Steering vector = v * valence_axis + a * arousal_axis
        direction = (v * valence_axis.float() + a * arousal_axis.float())
        # Normalize blended direction
        direction = direction / (direction.norm() + 1e-8)
        # Cast back to model dtype for steering hook compatibility
        direction = direction.to(model.dtype)
        alpha = alpha_scale * (abs(v) + abs(a)) / 2.0

        response = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=NEUTRAL_PROMPT,
            layer_names=[TARGET_LAYER],
            direction=direction,
            alpha=alpha,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
        )
        results[label] = {
            "valence": v,
            "arousal": a,
            "alpha": alpha,
            "response": response.strip(),
        }
        print(f"\n--- {label} (v={v}, a={a}, α={alpha:.2f}) ---")
        print(response.strip())

    return results


def main():
    print("=" * 60)
    print("Experiment 07 — Multi-Emotion Spectrum Validation")
    print("=" * 60)

    # Load model
    print(f"\nLoading {MODEL_NAME} on {DEVICE}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
    model.eval()
    print(f"  {len(get_layer_names(model))} layers, hidden={model.config.hidden_size}")

    # Load per-emotion directions
    directions = {}
    for emo in EMOTIONS:
        path = PROJECT_ROOT / "outputs" / "directions" / f"{emo}_direction.pt"
        directions[emo] = torch.load(path, weights_only=True)[TARGET_LAYER].to(DEVICE)
        print(f"  Loaded {emo} direction (norm={directions[emo].norm():.3f})")

    # Load valence/arousal axes
    axes_path = PROJECT_ROOT / "outputs" / "directions" / f"valence_arousal_axes_{TARGET_LAYER.replace('.', '_')}.pt"
    axes = torch.load(axes_path, weights_only=True).to(DEVICE)
    print(f"  Loaded valence/arousal axes (shape={axes.shape})")

    # Test 1: Per-emotion steering
    per_emotion_results = test_per_emotion_steering(model, tokenizer, directions)

    # Test 2: 2D plane geometry
    coords = test_2d_plane_geometry(axes, directions)

    # Test 3: Blended 2D steering
    blended_results = test_blended_steering(model, tokenizer, axes[0], axes[1])

    # Save results
    out = {
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "per_emotion_steering": per_emotion_results,
        "plane_geometry": coords,
        "blended_steering": blended_results,
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print("Experiment 07 complete.")


if __name__ == "__main__":
    main()
