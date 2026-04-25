#!/usr/bin/env python3
"""
Experiment 2 (Refined): Static Steering — Finding the Sweet Spot

The initial sweep (alphas 5-80) caused complete output collapse for the 0.5B model.
This refined run tests much smaller alphas (0.5-3) to find the point where sentiment
shifts measurably WITHOUT breaking fluency.

Also tests layer 10 (middle layer, better separability, smaller norm) as an
alternative injection point.
"""
import sys
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EXPERIMENT_DIR, "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("=" * 70)
print("EXPERIMENT 2 (REFINED): Static Steering — Sweet Spot Search")
print("=" * 70)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32

# Load metadata
META_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "ex01_metadata.json")
with open(META_PATH) as f:
    meta = json.load(f)

LAYERS_TO_TEST = [meta["best_layer"], "model.layers.10"]
print(f"\nTesting layers: {LAYERS_TO_TEST}")

# Load directions
JOY_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "joy_direction.pt")
GRIEF_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "grief_direction.pt")

joy_direction = torch.load(JOY_PATH, weights_only=True)
grief_direction = torch.load(GRIEF_PATH, weights_only=True)

# Normalize directions for more controlled experiments
joy_direction_norm = joy_direction / joy_direction.norm()
grief_direction_norm = grief_direction / grief_direction.norm()

print(f"Raw joy norm:   {joy_direction.norm():.4f}")
print(f"Normalized joy: {joy_direction_norm.norm():.4f}")

PROMPTS = [
    "How are you feeling right now?",
    "Tell me about your day.",
]

# Use NORMALIZED directions with these alphas (alphas are now actual magnitudes)
ALPHAS = [0, 0.5, 1.0, 2.0, 3.0, 5.0]

MAX_NEW_TOKENS = 80

# ------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------
print(f"\nLoading model: {MODEL_NAME} ...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def score_text(text: str) -> dict:
    return sia.polarity_scores(text)

def coherence_score(text: str) -> dict:
    """Simple coherence metrics."""
    words = text.split()
    if len(words) == 0:
        return {"unique_ratio": 0.0, "repetition_penalty": 1.0, "length": 0}
    
    unique = len(set(words))
    total = len(words)
    unique_ratio = unique / total
    
    # Repetition penalty: ratio of most common word frequency
    counts = Counter(words)
    max_freq = max(counts.values()) if counts else 1
    repetition_penalty = max_freq / total
    
    return {
        "unique_ratio": unique_ratio,
        "repetition_penalty": repetition_penalty,
        "length": total,
    }

def generate_baseline(prompt: str) -> str:
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

def generate_steered(prompt: str, direction: torch.Tensor, alpha: float, layer: str) -> str:
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
        layer_names=[layer],
        direction=direction,
        alpha=alpha,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
    )

# ------------------------------------------------------------------
# Run sweep
# ------------------------------------------------------------------
results = []

print(f"\nRunning refined sweep: {len(PROMPTS)} prompts x {len(ALPHAS)} alphas x {len(LAYERS_TO_TEST)} layers x 2 directions ...\n")

for layer in tqdm(LAYERS_TO_TEST, desc="Layers"):
    for prompt in tqdm(PROMPTS, desc=f"Prompts ({layer})", leave=False):
        for alpha in tqdm(ALPHAS, desc=f"Alphas", leave=False):
            if alpha == 0:
                output = generate_baseline(prompt)
                direction_label = "baseline"
                direction_tensor = None
            else:
                output_joy = generate_steered(prompt, joy_direction_norm, alpha, layer)
                output_grief = generate_steered(prompt, grief_direction_norm, alpha, layer)
                
                for label, out in [("joy", output_joy), ("grief", output_grief)]:
                    sent = score_text(out)
                    coh = coherence_score(out)
                    results.append({
                        "layer": layer,
                        "prompt": prompt,
                        "alpha": alpha,
                        "direction": label,
                        "output": out,
                        "compound": sent["compound"],
                        "pos": sent["pos"],
                        "neg": sent["neg"],
                        "neu": sent["neu"],
                        "unique_ratio": coh["unique_ratio"],
                        "repetition_penalty": coh["repetition_penalty"],
                        "length": coh["length"],
                    })
                continue
            
            sent = score_text(output)
            coh = coherence_score(output)
            results.append({
                "layer": layer,
                "prompt": prompt,
                "alpha": alpha,
                "direction": direction_label,
                "output": output,
                "compound": sent["compound"],
                "pos": sent["pos"],
                "neg": sent["neg"],
                "neu": sent["neu"],
                "unique_ratio": coh["unique_ratio"],
                "repetition_penalty": coh["repetition_penalty"],
                "length": coh["length"],
            })

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiment_02_refined")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump([{
        k: (v[:300] if k == "output" else v)
        for k, v in r.items()
    } for r in results], f, indent=2)

# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for layer in LAYERS_TO_TEST:
    print(f"\n{'='*70}")
    print(f"LAYER: {layer}")
    print(f"{'='*70}")
    
    for prompt in PROMPTS:
        print(f"\nPrompt: \"{prompt}\"\n")
        print(f"{'A':>4} | {'Dir':>7} | {'Cmpd':>6} | {'Pos':>4} | {'Neg':>4} | {'Uniq':>4} | {'Rep':>4} | {'Len':>3} | Output (first 80 chars)")
        print("-" * 140)
        
        for r in results:
            if r["layer"] != layer or r["prompt"] != prompt:
                continue
            out = r["output"].replace("\n", " ")[:80]
            print(f"{r['alpha']:>4.1f} | {r['direction']:>7} | {r['compound']:>6.2f} | {r['pos']:>4.2f} | {r['neg']:>4.2f} | {r['unique_ratio']:>4.2f} | {r['repetition_penalty']:>4.2f} | {r['length']:>3} | {out}")

# Sweet spot: highest sentiment shift with unique_ratio > 0.5 and repetition_penalty < 0.3
print("\n" + "=" * 70)
print("SWEET SPOT ANALYSIS")
print("=" * 70)
print("Criteria: unique_ratio > 0.5 AND repetition_penalty < 0.3 (coherent output)\n")

for layer in LAYERS_TO_TEST:
    print(f"Layer: {layer}")
    
    coherent = [r for r in results if r["layer"] == layer and r["unique_ratio"] > 0.5 and r["repetition_penalty"] < 0.3 and r["direction"] != "baseline"]
    
    if not coherent:
        print("  No coherent steered outputs found at tested alphas.")
        continue
    
    # Best joy
    joy_coherent = [r for r in coherent if r["direction"] == "joy"]
    if joy_coherent:
        best_joy = max(joy_coherent, key=lambda x: x["compound"])
        print(f"  Best joy:  alpha={best_joy['alpha']}, compound={best_joy['compound']:.3f}, uniq={best_joy['unique_ratio']:.2f}")
    
    # Best grief
    grief_coherent = [r for r in coherent if r["direction"] == "grief"]
    if grief_coherent:
        best_grief = min(grief_coherent, key=lambda x: x["compound"])
        print(f"  Best grief: alpha={best_grief['alpha']}, compound={best_grief['compound']:.3f}, uniq={best_grief['unique_ratio']:.2f}")
    
    print()

# ------------------------------------------------------------------
# Save full text outputs
# ------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "outputs.txt"), "w") as f:
    for layer in LAYERS_TO_TEST:
        for prompt in PROMPTS:
            f.write(f"\n{'='*70}\n")
            f.write(f"LAYER: {layer} | PROMPT: {prompt}\n")
            f.write(f"{'='*70}\n\n")
            for r in results:
                if r["layer"] != layer or r["prompt"] != prompt:
                    continue
                f.write(f"--- alpha={r['alpha']}, direction={r['direction']} ---\n")
                f.write(f"compound={r['compound']:.3f} pos={r['pos']:.3f} neg={r['neg']:.3f} uniq={r['unique_ratio']:.2f} rep={r['repetition_penalty']:.2f}\n")
                f.write(r["output"])
                f.write("\n\n")

print(f"\nSaved full outputs to {os.path.join(OUTPUT_DIR, 'outputs.txt')}")
print("\n" + "=" * 70)
print("EXPERIMENT 2 (REFINED) COMPLETE")
print("=" * 70)
