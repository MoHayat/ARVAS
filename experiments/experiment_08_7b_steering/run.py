"""
Experiment 08 — 7B Model Steering Validation

Tests the full 2D valence-arousal steering pipeline on a larger model.
Validates that:
1. The 7B model loads successfully on MPS + fp16 within 48GB unified memory.
2. Per-emotion directions produce distinct response tones.
3. 2D blended steering (valence, arousal) produces nuanced emotional shifts.
4. The model's template entrenchment is overcome with appropriate prompts.

Usage:
    # First ensure 7B directions exist (run separately if needed):
    python src/emotion_extraction.py --model Qwen/Qwen2.5-7B-Instruct \
        --stories data/emotion_stories.json --output outputs/directions_7b \
        --device mps --torch_dtype float16 --layers model.layers.20

    # Then run this experiment:
    cd /Users/mohayat/projects/KH/ARVAS
    source venv/bin/activate
    python experiments/experiment_08_7b_steering/run.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import torch
from activation_utils import load_model_and_tokenizer, get_layer_names
from steering import generate_with_steering, compute_2d_direction

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEVICE = "mps"
DTYPE = torch.float16
MAX_NEW_TOKENS = 120
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment_08"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["joy", "excitement", "calm", "boredom", "sadness", "fear", "anger", "disgust"]

# Neutral prompts designed to be emotionally malleable on 7B
# Note: 7B models have stronger "AI assistant" priors. We use creative/open-ended 
# prompts that resist templating — no "how are you" or "what's on your mind".
NEUTRAL_PROMPTS = [
    "Write a short poem about a thunderstorm. Just the poem, no introduction.",
    "Describe the taste of a food you loved as a child. One paragraph, sensory details.",
    "Tell me about a memory that surfaces when you hear rain on the roof. Be brief.",
]

# Steering coefficients tuned for 7B (normalized directions)
# 7B representations are richer but more resistant; alphas need to be higher
# to overcome template entrenchment. Start conservative and note if higher is needed.
ALPHA_PER_EMOTION = {
    "joy": 8.0,
    "excitement": 8.0,
    "calm": 6.0,
    "boredom": 6.0,
    "sadness": 8.0,
    "fear": 8.0,
    "anger": 7.0,
    "disgust": 7.0,
}

# 2D test points
BLENDED_TESTS = [
    ("High Joy (Q1)", 1.0, 1.0, 8.0),
    ("Angry (Q4)", -1.0, 1.0, 7.0),
    ("Sad (Q3)", -1.0, -0.5, 6.0),
    ("Calm (Q2)", 0.5, -1.0, 6.0),
    ("Neutral", 0.0, 0.0, 0.0),
]


def test_per_emotion_steering(model, tokenizer, directions, target_layer):
    """Apply each emotion direction individually across multiple prompts."""
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
            print(f"\nPrompt {i+1}: {r[:200]}...")
    return results


def test_blended_2d_steering(model, tokenizer, valence_axis, arousal_axis, target_layer):
    """Test blended 2D steering across multiple prompts."""
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
            print(f"  [{i+1}] {r[:180]}...")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to test (default: 7B). Can also use 1.5B for faster iteration.")
    parser.add_argument("--directions-dir", default=None,
                        help="Directory containing emotion directions. Defaults to outputs/directions_7b for 7B model.")
    parser.add_argument("--target-layer", default=None,
                        help="Target layer for steering. Auto-detected as middle layer if not provided.")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 08 — 7B Model Steering Validation")
    print("=" * 60)

    # Determine directions directory
    if args.directions_dir:
        directions_dir = Path(args.directions_dir)
    elif "7B" in args.model:
        directions_dir = PROJECT_ROOT / "outputs" / "directions_7b"
    else:
        directions_dir = PROJECT_ROOT / "outputs" / "directions"

    if not directions_dir.exists():
        print(f"\nERROR: Directions directory not found: {directions_dir}")
        print("Run emotion extraction first:")
        print(f"  python src/emotion_extraction.py --model {args.model} --stories data/emotion_stories.json --output {directions_dir} --device {DEVICE} --torch_dtype float16")
        sys.exit(1)

    # Auto-detect target layer if not specified
    if args.target_layer:
        target_layer = args.target_layer
    else:
        # Quick load to count layers
        print("\nAuto-detecting optimal middle layer...")
        tmp_model, _ = load_model_and_tokenizer(args.model, device=DEVICE, torch_dtype=DTYPE)
        n_layers = len(get_layer_names(tmp_model))
        target_layer = f"model.layers.{n_layers // 2}"
        print(f"  Model has {n_layers} layers → using {target_layer}")
        del tmp_model
        torch.mps.empty_cache()

    # Load model for real
    print(f"\nLoading {args.model} on {DEVICE} with {DTYPE}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=DEVICE, torch_dtype=DTYPE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = len(get_layer_names(model))
    print(f"  {n_layers} layers, hidden={model.config.hidden_size}")

    # Load directions
    directions = {}
    for emo in EMOTIONS:
        path = directions_dir / f"{emo}_direction.pt"
        if not path.exists():
            print(f"  WARNING: Missing direction for {emo} at {path}")
            continue
        # Load dict of layer->tensor, pick target_layer
        d = torch.load(path, weights_only=True)
        if target_layer in d:
            directions[emo] = d[target_layer].to(DEVICE)
        else:
            # Fallback: pick first available layer
            first_layer = list(d.keys())[0]
            directions[emo] = d[first_layer].to(DEVICE)
            print(f"  {emo}: target layer not found, using {first_layer}")
        print(f"  Loaded {emo} direction (norm={directions[emo].norm():.3f})")

    # Load valence/arousal axes
    axes_path = directions_dir / f"valence_arousal_axes_{target_layer.replace('.', '_')}.pt"
    if axes_path.exists():
        axes = torch.load(axes_path, weights_only=True).to(DEVICE)
        valence_axis = axes[0]
        arousal_axis = axes[1]
        print(f"  Loaded 2D axes from {axes_path.name}")
    else:
        # Fallback: glob for any axes file
        alt = list(directions_dir.glob("valence_arousal_axes_*.pt"))
        if alt:
            axes = torch.load(alt[0], weights_only=True).to(DEVICE)
            valence_axis = axes[0]
            arousal_axis = axes[1]
            print(f"  Loaded fallback axes: {alt[0].name}")
        else:
            print("  ERROR: No valence/arousal axes found. Run extraction first.")
            sys.exit(1)

    # Test 1: Per-emotion steering
    per_emotion_results = test_per_emotion_steering(model, tokenizer, directions, target_layer)

    # Test 2: Blended 2D steering
    blended_results = test_blended_2d_steering(model, tokenizer, valence_axis, arousal_axis, target_layer)

    # Save results
    out = {
        "model": args.model,
        "target_layer": target_layer,
        "directions_dir": str(directions_dir),
        "per_emotion_steering": per_emotion_results,
        "blended_2d_steering": blended_results,
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print("Experiment 08 complete.")


if __name__ == "__main__":
    main()
