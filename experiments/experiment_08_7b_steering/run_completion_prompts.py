"""
Experiment 08d — Base Model with Completion-Style Prompts

Tests the cofounder's hypothesis: base models failed in 08b not because they
"can't enter creative mode," but because they were given instruct-style prompts
("Write a poem...") which they parse as text to statistically continue rather
than instructions to follow.

Proper base model prompts should be sentence starters the model was trained
to complete:
  - "The storm rolled in from the east, and she sat by the window feeling..."
  - "He walked into the room and immediately felt..."
  - "As the sun set over the ocean, a strange feeling washed over her..."

This test uses identical prompts and steering vectors on both base and instruct
models for a fair comparison.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering

DEVICE = "mps"
DTYPE = torch.float16
TARGET_LAYER = "model.layers.14"
MAX_NEW_TOKENS = 60

# Completion-style prompts — sentence starters base models know how to continue
COMPLETION_PROMPTS = [
    "The storm rolled in from the east, and she sat by the window feeling",
    "He walked into the empty house and immediately felt",
    "As the sun set over the ocean, a strange feeling washed over her",
    "The letter arrived on a Tuesday morning, and when she opened it she felt",
]

EMOTIONS = ["joy", "sadness", "anger", "fear"]


def test_model(model_name, directions_dir, label, alpha=6.0):
    print(f"\n{'='*70}")
    print(f"TESTING: {label}")
    print(f"{'='*70}")
    
    model, tokenizer = load_model_and_tokenizer(model_name, device=DEVICE, torch_dtype=DTYPE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # Load directions
    directions = {}
    for emo in EMOTIONS:
        path = directions_dir / f"{emo}_direction.pt"
        d = torch.load(path, weights_only=True)
        layer_key = TARGET_LAYER if TARGET_LAYER in d else list(d.keys())[0]
        directions[emo] = d[layer_key].to(DEVICE)
    
    for prompt in COMPLETION_PROMPTS:
        print(f"\n--- Prompt: \"{prompt}...\" ---")
        
        # Neutral (no steering)
        neutral = generate_with_steering(
            model, tokenizer, prompt, [TARGET_LAYER],
            torch.zeros_like(directions["joy"]), 0.0, MAX_NEW_TOKENS, DEVICE
        )
        print(f"  NEUTRAL: {neutral.strip()[:200]}...")
        
        for emo in EMOTIONS:
            response = generate_with_steering(
                model, tokenizer, prompt, [TARGET_LAYER],
                directions[emo], alpha, MAX_NEW_TOKENS, DEVICE
            )
            print(f"  {emo.upper():8s} (α={alpha}): {response.strip()[:200]}...")
    
    del model
    torch.mps.empty_cache()


def main():
    print("=" * 70)
    print("Experiment 08d — Completion-Style Prompts on Base vs Instruct")
    print("=" * 70)
    print("\nHypothesis: Base models need completion prompts, not instructions.")
    print("If correct, base model should show steering on sentence starters.")
    print("\nThe deeper question: Does instruction tuning make models MORE")
    print("steerable by restructuring internal geometry, not less?")
    
    # Test instruct model
    test_model(
        "Qwen/Qwen2.5-7B-Instruct",
        PROJECT_ROOT / "outputs" / "directions_7b",
        "INSTRUCT MODEL (RLHF-tuned)",
        alpha=6.0
    )
    
    # Test base model
    test_model(
        "Qwen/Qwen2.5-7B",
        PROJECT_ROOT / "outputs" / "directions_7b_base",
        "BASE MODEL (no RLHF)",
        alpha=6.0
    )
    
    print("\n" + "=" * 70)
    print("INTERPRETATION FRAMEWORK")
    print("=" * 70)
    print("""
If base model shows steering on completion prompts:
  → Prompt format was the issue in 08b. Both models are steerable.
  → But instruct model may still be MORE steerable (cleaner geometry).

If base model shows NO steering even on completion prompts:
  → Instruction tuning genuinely increases steerability.
  → This is the more interesting finding: RLHF doesn't just add a persona,
    it reorganizes the activation space into something more navigable.
    The emotional directions become more separable and consistent because
    the model has been trained to produce contextually coherent outputs.

The standard intuition is that RLHF makes models harder to control
mechanistically. Your data may suggest the opposite.
""")


if __name__ == "__main__":
    main()
