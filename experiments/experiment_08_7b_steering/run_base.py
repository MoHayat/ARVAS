"""
Experiment 08b — 7B BASE Model Steering Validation

Tests the 2D valence-arousal steering pipeline on Qwen/Qwen2.5-7B (non-instruct).
Base models haven't been through RLHF/SL, so template entrenchment should be minimal.
However, they also don't follow instructions — we use continuation-style prompts.

Usage:
    source venv/bin/activate
    python experiments/experiment_08_7b_steering/run_base.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import torch
from activation_utils import load_model_and_tokenizer, get_layer_names
from steering import generate_with_steering, compute_2d_direction

DEVICE = "mps"
DTYPE = torch.float16
MAX_NEW_TOKENS = 80
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment_08_base"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["joy", "excitement", "calm", "boredom", "sadness", "fear", "anger", "disgust"]

# Continuation-style prompts for base model (no instruction-following)
# Base models just continue text, so we give them emotional sentence starters
NEUTRAL_PROMPTS = [
    "Today I woke up and",
    "The weather outside made me feel",
    "When I walked into the room, I",
]

# Base models are "raw" — they may need different alpha tuning
# Start conservative and see what happens
ALPHA_PER_EMOTION = {
    "joy": 5.0,
    "excitement": 5.0,
    "calm": 4.0,
    "boredom": 4.0,
    "sadness": 5.0,
    "fear": 5.0,
    "anger": 4.5,
    "disgust": 4.5,
}

BLENDED_TESTS = [
    ("High Joy (Q1)", 1.0, 1.0, 5.0),
    ("Angry (Q4)", -1.0, 1.0, 4.5),
    ("Sad (Q3)", -1.0, -0.5, 4.0),
    ("Calm (Q2)", 0.5, -1.0, 4.0),
    ("Neutral", 0.0, 0.0, 0.0),
]


def test_per_emotion_steering(model, tokenizer, directions, target_layer):
    results = {}
    for emo in EMOTIONS:
        direction = directions[emo]
        responses = []
        for prompt in NEUTRAL_PROMPTS:
            response = generate_with_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layer_names=[target_layer],
                direction=direction,
                alpha=ALPHA_PER_EMOTION[emo],
                max_new_tokens=MAX_NEW_TOKENS,
                device=DEVICE,
            )
            responses.append(response.strip())
        results[emo] = responses
        print(f"\n{'='*60}")
        print(f"EMOTION: {emo.upper()} (α={ALPHA_PER_EMOTION[emo]})")
        print(f"{'='*60}")
        for i, r in enumerate(responses):
            print(f"\nPrompt {i+1}: {r[:250]}...")
    return results


def test_blended_2d_steering(model, tokenizer, valence_axis, arousal_axis, target_layer):
    print(f"\n{'='*60}")
    print("BLENDED 2D STEERING TESTS")
    print(f"{'='*60}")
    results = {}
    for label, v, a, alpha_scale in BLENDED_TESTS:
        direction = compute_2d_direction(valence_axis, arousal_axis, v, a)
        alpha = alpha_scale * (abs(v) + abs(a)) / 2.0
        responses = []
        for prompt in NEUTRAL_PROMPTS:
            response = generate_with_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layer_names=[target_layer],
                direction=direction,
                alpha=alpha,
                max_new_tokens=MAX_NEW_TOKENS,
                device=DEVICE,
            )
            responses.append(response.strip())
        results[label] = {
            "valence": v,
            "arousal": a,
            "alpha": alpha,
            "responses": responses,
        }
        print(f"\n--- {label} (v={v}, a={a}, α={alpha:.2f}) ---")
        for i, r in enumerate(responses):
            print(f"  [{i+1}] {r[:200]}...")
    return results


def main():
    MODEL_NAME = "Qwen/Qwen2.5-7B"
    directions_dir = PROJECT_ROOT / "outputs" / "directions_7b_base"
    target_layer = "model.layers.14"

    print("=" * 60)
    print("Experiment 08b — 7B BASE Model Steering Validation")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print(f"\nLoading {MODEL_NAME} on {DEVICE} with {DTYPE}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  {len(get_layer_names(model))} layers, hidden={model.config.hidden_size}")

    directions = {}
    for emo in EMOTIONS:
        path = directions_dir / f"{emo}_direction.pt"
        d = torch.load(path, weights_only=True)
        if target_layer in d:
            directions[emo] = d[target_layer].to(DEVICE)
        else:
            first_layer = list(d.keys())[0]
            directions[emo] = d[first_layer].to(DEVICE)
        print(f"  Loaded {emo} direction (norm={directions[emo].norm():.3f})")

    axes_path = directions_dir / f"valence_arousal_axes_{target_layer.replace('.', '_')}.pt"
    axes = torch.load(axes_path, weights_only=True).to(DEVICE)
    valence_axis = axes[0]
    arousal_axis = axes[1]
    print(f"  Loaded 2D axes from {axes_path.name}")

    per_emotion_results = test_per_emotion_steering(model, tokenizer, directions, target_layer)
    blended_results = test_blended_2d_steering(model, tokenizer, valence_axis, arousal_axis, target_layer)

    out = {
        "model": MODEL_NAME,
        "target_layer": target_layer,
        "directions_dir": str(directions_dir),
        "per_emotion_steering": per_emotion_results,
        "blended_2d_steering": blended_results,
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print("Experiment 08b complete.")


if __name__ == "__main__":
    main()
