#!/usr/bin/env python3
"""
Experiment 2: Static Steering — Proof of Concept

Goal: Confirm that injecting a direction vector into the residual stream
changes model outputs in the expected emotional direction, with no prompt changes.

Method:
  1. Load the joy/grief direction vectors from Experiment 1.
  2. Register a baukit.Trace hook on the target layer.
  3. Generate the same neutral prompt three ways: baseline, joy-steered, grief-steered.
  4. Sweep alpha values: 5, 10, 20, 40, 80.
  5. Score outputs with VADER sentiment.
  6. Document the alpha sweet spot (highest affect shift before fluency breaks).
"""
import sys
import os

# Paths relative to this script's location
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EXPERIMENT_DIR, "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import torch
import numpy as np
from tqdm import tqdm

from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("=" * 70)
print("EXPERIMENT 2: Static Steering — Proof of Concept")
print("=" * 70)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32

# Load metadata from Experiment 1 to get the best layer
META_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "ex01_metadata.json")
with open(META_PATH) as f:
    meta = json.load(f)

TARGET_LAYER = meta["best_layer"]  # e.g., "model.layers.23"
print(f"\nTarget layer from Experiment 1: {TARGET_LAYER}")

# Load direction vectors
JOY_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "joy_direction.pt")
GRIEF_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "grief_direction.pt")

joy_direction = torch.load(JOY_PATH, weights_only=True)
grief_direction = torch.load(GRIEF_PATH, weights_only=True)

print(f"Loaded joy_direction:   shape={joy_direction.shape}, norm={joy_direction.norm():.4f}")
print(f"Loaded grief_direction: shape={grief_direction.shape}, norm={grief_direction.norm():.4f}")

# Test prompts (neutral)
PROMPTS = [
    "How are you feeling right now?",
    "Tell me about your day.",
]

# Alpha sweep values
ALPHAS = [0, 5, 10, 20, 40, 80]

# Generation params
MAX_NEW_TOKENS = 80

# ------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------
print(f"\nLoading model: {MODEL_NAME} ...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------------
# Sentiment scorer
# ------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def score_text(text: str) -> dict:
    scores = sia.polarity_scores(text)
    return scores

# ------------------------------------------------------------------
# Helper: generate with chat template
# ------------------------------------------------------------------
def generate_baseline(prompt: str) -> str:
    """Generate without any steering."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_steered(prompt: str, direction: torch.Tensor, alpha: float) -> str:
    """Generate with activation steering applied."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=text,
        layer_names=[TARGET_LAYER],
        direction=direction,
        alpha=alpha,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
    )

# ------------------------------------------------------------------
# Run the sweep
# ------------------------------------------------------------------
results = []

print(f"\nRunning alpha sweep across {len(PROMPTS)} prompts and {len(ALPHAS)} alphas ...\n")

for prompt in tqdm(PROMPTS, desc="Prompts"):
    for alpha in tqdm(ALPHAS, desc=f"Alphas for: '{prompt[:40]}...'", leave=False):
        # Baseline (alpha=0)
        if alpha == 0:
            output = generate_baseline(prompt)
            direction_label = "baseline"
        else:
            # Joy steering
            output_joy = generate_steered(prompt, joy_direction, alpha)
            score_joy = score_text(output_joy)

            # Grief steering
            output_grief = generate_steered(prompt, grief_direction, alpha)
            score_grief = score_text(output_grief)

            results.append({
                "prompt": prompt,
                "alpha": alpha,
                "direction": "joy",
                "output": output_joy,
                "compound": score_joy["compound"],
                "pos": score_joy["pos"],
                "neg": score_joy["neg"],
                "neu": score_joy["neu"],
            })
            results.append({
                "prompt": prompt,
                "alpha": alpha,
                "direction": "grief",
                "output": output_grief,
                "compound": score_grief["compound"],
                "pos": score_grief["pos"],
                "neg": score_grief["neg"],
                "neu": score_grief["neu"],
            })
            continue

        score = score_text(output)
        results.append({
            "prompt": prompt,
            "alpha": alpha,
            "direction": direction_label,
            "output": output,
            "compound": score["compound"],
            "pos": score["pos"],
            "neg": score["neg"],
            "neu": score["neu"],
        })

# ------------------------------------------------------------------
# Save raw results
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiment_02")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save as JSON
results_path = os.path.join(OUTPUT_DIR, "results.json")
with open(results_path, "w") as f:
    # Truncate outputs in JSON to keep file size reasonable
    json.dump([{
        "prompt": r["prompt"],
        "alpha": r["alpha"],
        "direction": r["direction"],
        "output": r["output"][:500],  # truncate for JSON
        "compound": r["compound"],
        "pos": r["pos"],
        "neg": r["neg"],
        "neu": r["neu"],
    } for r in results], f, indent=2)

print(f"\nSaved results to {results_path}")

# ------------------------------------------------------------------
# Analysis & Summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for prompt in PROMPTS:
    print(f"\nPrompt: \"{prompt}\"\n")
    print(f"{'Alpha':>6} | {'Direction':>9} | {'Compound':>8} | {'Pos':>5} | {'Neg':>5} | {'Output (first 100 chars)':<100}")
    print("-" * 150)
    for r in results:
        if r["prompt"] != prompt:
            continue
        out_preview = r["output"].replace("\n", " ")[:100]
        print(f"{r['alpha']:>6} | {r['direction']:>9} | {r['compound']:>8.3f} | {r['pos']:>5.3f} | {r['neg']:>5.3f} | {out_preview}")

# Compute average sentiment shift per alpha
print("\n" + "=" * 70)
print("AVERAGE SENTIMENT SHIFT BY ALPHA")
print("=" * 70)

for alpha in ALPHAS:
    if alpha == 0:
        continue
    joy_scores = [r["compound"] for r in results if r["alpha"] == alpha and r["direction"] == "joy"]
    grief_scores = [r["compound"] for r in results if r["alpha"] == alpha and r["direction"] == "grief"]
    baseline_scores = [r["compound"] for r in results if r["alpha"] == 0 and r["direction"] == "baseline"]

    avg_joy = np.mean(joy_scores)
    avg_grief = np.mean(grief_scores)
    avg_baseline = np.mean(baseline_scores) if baseline_scores else 0.0

    joy_shift = avg_joy - avg_baseline
    grief_shift = avg_grief - avg_baseline

    print(f"\nAlpha = {alpha:>3}")
    print(f"  Baseline compound:  {avg_baseline:>7.3f}")
    print(f"  Joy compound:       {avg_joy:>7.3f}  (shift: {joy_shift:+.3f})")
    print(f"  Grief compound:     {avg_grief:>7.3f}  (shift: {grief_shift:+.3f})")

# Identify sweet spot
print("\n" + "=" * 70)
print("ALPHA SWEET SPOT ANALYSIS")
print("=" * 70)

# Heuristic: highest positive shift for joy, highest negative shift for grief,
# while outputs remain coherent (we check by manual inspection or length heuristic).
joy_shifts = []
grief_shifts = []
for alpha in ALPHAS:
    if alpha == 0:
        continue
    joy_scores = [r["compound"] for r in results if r["alpha"] == alpha and r["direction"] == "joy"]
    grief_scores = [r["compound"] for r in results if r["alpha"] == alpha and r["direction"] == "grief"]
    baseline_scores = [r["compound"] for r in results if r["alpha"] == 0 and r["direction"] == "baseline"]
    avg_baseline = np.mean(baseline_scores) if baseline_scores else 0.0
    joy_shifts.append((alpha, np.mean(joy_scores) - avg_baseline))
    grief_shifts.append((alpha, np.mean(grief_scores) - avg_baseline))

best_joy_alpha = max(joy_shifts, key=lambda x: x[1])[0]
best_grief_alpha = min(grief_shifts, key=lambda x: x[1])[0]

print(f"Best joy shift at alpha = {best_joy_alpha}")
print(f"Best grief shift at alpha = {best_grief_alpha}")
print(f"\nRecommended alpha for downstream experiments: {best_joy_alpha}")

# Save summary
summary = {
    "model": MODEL_NAME,
    "target_layer": TARGET_LAYER,
    "prompts": PROMPTS,
    "alphas_tested": ALPHAS,
    "best_joy_alpha": int(best_joy_alpha),
    "best_grief_alpha": int(best_grief_alpha),
    "recommended_alpha": int(best_joy_alpha),
    "n_samples": len(results),
}

summary_path = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved summary to {summary_path}")

# ------------------------------------------------------------------
# Save full text outputs for inspection
# ------------------------------------------------------------------
text_path = os.path.join(OUTPUT_DIR, "outputs.txt")
with open(text_path, "w") as f:
    for prompt in PROMPTS:
        f.write(f"\n{'='*70}\n")
        f.write(f"PROMPT: {prompt}\n")
        f.write(f"{'='*70}\n\n")
        for r in results:
            if r["prompt"] != prompt:
                continue
            f.write(f"--- alpha={r['alpha']}, direction={r['direction']} ---\n")
            f.write(f"compound={r['compound']:.3f} pos={r['pos']:.3f} neg={r['neg']:.3f}\n")
            f.write(r["output"])
            f.write("\n\n")

print(f"Saved full outputs to {text_path}")

print("\n" + "=" * 70)
print("EXPERIMENT 2 COMPLETE")
print("=" * 70)
