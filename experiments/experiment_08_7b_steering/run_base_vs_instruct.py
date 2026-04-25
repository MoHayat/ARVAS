"""
Experiment 08c — 7B Base vs Instruct Comparison

Direct comparison of steering on Qwen2.5-7B (base) vs Qwen2.5-7B-Instruct.
Tests the hypothesis that RLHF/SL causes template entrenchment that blocks steering.

Finding: Base models have NO template entrenchment, but they also have NO coherence.
They fall into repetitive pre-training data patterns and cannot generate novel emotional text.
The instruct model's "template mode" is actually a feature — it enables coherent generation.
Steering works when we break the model out of template mode via creative prompts.
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

# Same creative prompt that worked on instruct model
CREATIVE_PROMPT = "Thunder roars, a"

def test_model(model_name, directions_dir, label):
    print(f"\n{'='*70}")
    print(f"TESTING: {label}")
    print(f"{'='*70}")
    
    model, tokenizer = load_model_and_tokenizer(model_name, device=DEVICE, torch_dtype=DTYPE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # Load joy and anger directions
    joy_path = directions_dir / "joy_direction.pt"
    anger_path = directions_dir / "anger_direction.pt"
    
    joy_dir = torch.load(joy_path, weights_only=True)[TARGET_LAYER].to(DEVICE)
    anger_dir = torch.load(anger_path, weights_only=True)[TARGET_LAYER].to(DEVICE)
    
    print(f"\n--- NEUTRAL (no steering) ---")
    neutral = generate_with_steering(
        model, tokenizer, CREATIVE_PROMPT, [TARGET_LAYER], 
        torch.zeros_like(joy_dir), 0.0, 40, DEVICE
    )
    print(neutral.strip()[:300])
    
    print(f"\n--- JOY (α=8.0) ---")
    joy = generate_with_steering(
        model, tokenizer, CREATIVE_PROMPT, [TARGET_LAYER],
        joy_dir, 8.0, 40, DEVICE
    )
    print(joy.strip()[:300])
    
    print(f"\n--- ANGER (α=7.0) ---")
    anger = generate_with_steering(
        model, tokenizer, CREATIVE_PROMPT, [TARGET_LAYER],
        anger_dir, 7.0, 40, DEVICE
    )
    print(anger.strip()[:300])
    
    del model
    torch.mps.empty_cache()


def main():
    print("=" * 70)
    print("Experiment 08c — Base vs Instruct Steering Comparison")
    print("=" * 70)
    print("\nHypothesis: Base model (no RLHF) will show stronger steering")
    print("because there's no template entrenchment blocking it.")
    print("\nActual finding: Base model falls into pre-training data loops")
    print("and cannot generate coherent novel text. Instruct model CAN")
    print("generate novel text, and steering is visible on creative prompts.")
    
    # Test instruct model
    test_model(
        "Qwen/Qwen2.5-7B-Instruct",
        PROJECT_ROOT / "outputs" / "directions_7b",
        "INSTRUCT MODEL (RLHF-tuned)"
    )
    
    # Test base model
    test_model(
        "Qwen/Qwen2.5-7B",
        PROJECT_ROOT / "outputs" / "directions_7b_base",
        "BASE MODEL (no RLHF)"
    )
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The instruct model generates coherent, steerable text on creative prompts.
The base model generates repetitive, non-steerable text from pre-training.

RLHF doesn't BLOCK steering — it ENABLES the coherent generation that
makes steering meaningful. The "template entrenchment" is a side effect
of alignment, but the solution is prompt design, not removing alignment.

Key insight: Steering vectors need a "creative surface" to express themselves.
Base models have no such surface — they only regurgitate training data.
Instruct models have the surface, and we can access it with the right prompts.
""")

if __name__ == "__main__":
    main()
