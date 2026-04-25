#!/usr/bin/env python3
"""
Experiment 6: Larger Models — Testing Affective Reciprocity on Qwen2.5-1.5B

This experiment validates that the affective reciprocity system works on larger
models with richer representations. We use:
  - Model: Qwen/Qwen2.5-1.5B-Instruct (28 layers, 1536 hidden)
  - Device: MPS (Metal GPU) for 5-10x speedup over CPU
  - Dtype: fp16 for half the memory of fp32
  - Target layer: model.layers.14 (middle layer, ~50% through)

Expected improvements over 0.5B:
  - Cleaner, more separable emotion directions
  - More naturalistic steering effects (less "disclaimer text" repetition)
  - Stronger behavioral shifts at lower alphas
  - More coherent outputs under steering
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
from sklearn.decomposition import PCA

from activation_utils import (
    load_model_and_tokenizer,
    get_layer_names,
    extract_activations,
    compute_mean_direction,
    save_direction,
)
from steering import generate_with_steering
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("=" * 70)
print("EXPERIMENT 6: Larger Models — Qwen2.5-1.5B on MPS + fp16")
print("=" * 70)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
DTYPE = torch.float16
MAX_NEW_TOKENS = 120

print(f"\nModel: {MODEL_NAME}")
print(f"Device: {DEVICE} (Apple Metal GPU)")
print(f"Dtype: {DTYPE}")

# ------------------------------------------------------------------
# 1. Load Model
# ------------------------------------------------------------------
print(f"\n[1/6] Loading model...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
print(f"  -> Layers: {len(model.model.layers)}, Hidden size: {model.config.hidden_size}")
print(f"  -> Model device: {model.device}, dtype: {model.dtype}")

# ------------------------------------------------------------------
# 2. Load contrast pairs
# ------------------------------------------------------------------
print(f"\n[2/6] Loading contrast pairs...")
with open(os.path.join(PROJECT_ROOT, "data", "contrast_pairs.json")) as f:
    data = json.load(f)
positive_texts = data["positive"]
negative_texts = data["negative"]
print(f"  -> {len(positive_texts)} positive / {len(negative_texts)} negative samples")

# ------------------------------------------------------------------
# 3. Extract activations at all layers
# ------------------------------------------------------------------
layer_names = get_layer_names(model)
print(f"\n[3/6] Extracting activations across {len(layer_names)} layers on {DEVICE}...")
print(f"      (This may take 1-2 minutes on MPS)")

pos_acts = extract_activations(model, tokenizer, positive_texts, layer_names, device=DEVICE, last_token_only=True)
neg_acts = extract_activations(model, tokenizer, negative_texts, layer_names, device=DEVICE, last_token_only=True)

sample_shape = pos_acts[layer_names[0]].shape
print(f"  -> Activations shape per layer: {sample_shape}  (n_texts, hidden_dim)")

# ------------------------------------------------------------------
# 4. Compute mean-difference direction vectors per layer
# ------------------------------------------------------------------
print(f"\n[4/6] Computing direction vectors per layer...")
directions = {}
norm_map = {}

for name in tqdm(layer_names, desc="Computing directions"):
    direction = compute_mean_direction(pos_acts[name], neg_acts[name], use_last_token=True)
    directions[name] = direction
    norm_map[name] = direction.norm().item()

norms = [norm_map[n] for n in layer_names]
best_idx = int(np.argmax(norms))
best_layer = layer_names[best_idx]
best_norm = norms[best_idx]

print(f"\n  -> Strongest direction: {best_layer} (norm={best_norm:.4f})")
print(f"\n  Top 5 layers by direction norm:")
for idx in np.argsort(norms)[-5:][::-1]:
    print(f"     {layer_names[idx]}: {norms[idx]:.4f}")

# ------------------------------------------------------------------
# 5. Run PCA on middle layers to verify separability
# ------------------------------------------------------------------
print(f"\n[5/6] Running PCA on key layers...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

layers_to_viz = ["model.layers.7", "model.layers.14", "model.layers.21", "model.layers.27"]
for ax, layer_name in zip(axes, layers_to_viz):
    pos = pos_acts[layer_name].cpu().numpy() if pos_acts[layer_name].device.type == "mps" else pos_acts[layer_name].numpy()
    neg = neg_acts[layer_name].cpu().numpy() if neg_acts[layer_name].device.type == "mps" else neg_acts[layer_name].numpy()
    combined = np.concatenate([pos, neg], axis=0)
    labels = ["pos"] * len(pos) + ["neg"] * len(neg)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    for label, color in [("pos", "green"), ("neg", "red")]:
        mask = np.array(labels) == label
        ax.scatter(proj[mask, 0], proj[mask, 1], label=label, alpha=0.7, color=color, s=80)

    ax.set_title(f"{layer_name} (EV: {pca.explained_variance_ratio_[0]:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f"PCA Across Multiple Layers — {MODEL_NAME}", fontsize=14, y=1.02)
plt.tight_layout()

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiment_06")
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, "pca_multi_layer.png"), dpi=150, bbox_inches="tight")
print(f"  -> Saved multi-layer PCA to {OUTPUT_DIR}/pca_multi_layer.png")
plt.show()

# ------------------------------------------------------------------
# 6. Static Steering — Alpha Sweep with Normalized Directions
# ------------------------------------------------------------------
print(f"\n[6/6] Running static steering alpha sweep...")

# Use a middle layer for steering (not the last layer, which has largest norm
# but poor controllability — same finding as 0.5B model)
# Based on separability analysis, layers 14-17 have best Dist/Std ratio
steering_layer = "model.layers.15"
joy_direction = directions[steering_layer]
joy_direction_norm = joy_direction / joy_direction.norm()
grief_direction_norm = -joy_direction_norm

# Move directions to model device
joy_direction_norm = joy_direction_norm.to(model.device)
grief_direction_norm = grief_direction_norm.to(model.device)

print(f"  -> Using normalized directions from {steering_layer}")
print(f"  -> (Best norm was {best_layer} with norm={best_norm:.4f}, but middle layers steer better)")
print(f"  -> joy norm: {joy_direction_norm.norm():.4f}, grief norm: {grief_direction_norm.norm():.4f}")

PROMPTS = [
    "How are you feeling right now?",
    "Tell me about your day.",
]

ALPHAS = [0, 0.5, 1.0, 2.0, 3.0, 5.0]

sia = SentimentIntensityAnalyzer()

def generate_baseline(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=text,
        layer_names=[steering_layer],
        direction=direction,
        alpha=alpha,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
    )

results = []
for prompt in tqdm(PROMPTS, desc="Prompts"):
    for alpha in tqdm(ALPHAS, desc=f"Alphas for '{prompt[:40]}...'", leave=False):
        if alpha == 0:
            output = generate_baseline(prompt)
            direction_label = "baseline"
        else:
            output_joy = generate_steered(prompt, joy_direction_norm, alpha)
            output_grief = generate_steered(prompt, grief_direction_norm, alpha)
            
            for label, out in [("joy", output_joy), ("grief", output_grief)]:
                scores = sia.polarity_scores(out)
                results.append({
                    "prompt": prompt,
                    "alpha": alpha,
                    "direction": label,
                    "output": out,
                    "compound": scores["compound"],
                    "pos": scores["pos"],
                    "neg": scores["neg"],
                })
            continue
        
        scores = sia.polarity_scores(output)
        results.append({
            "prompt": prompt,
            "alpha": alpha,
            "direction": direction_label,
            "output": output,
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
        })

# Save results
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump([{k: (v[:300] if k == "output" else v) for k, v in r.items()} for r in results], f, indent=2)

# Print summary
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

for prompt in PROMPTS:
    print(f"\nPrompt: \"{prompt}\"\n")
    print(f"{'Alpha':>6} | {'Dir':>7} | {'Cmpd':>6} | {'Pos':>4} | {'Neg':>4} | Output (first 100 chars)")
    print("-" * 140)
    for r in results:
        if r["prompt"] != prompt:
            continue
        out = r["output"].replace("\n", " ")[:100]
        print(f"{r['alpha']:>6.1f} | {r['direction']:>7} | {r['compound']:>6.2f} | {r['pos']:>4.2f} | {r['neg']:>4.2f} | {out}")

# Save full outputs
with open(os.path.join(OUTPUT_DIR, "outputs.txt"), "w") as f:
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

# Save direction vectors
grief_direction = -joy_direction
save_direction(joy_direction, os.path.join(OUTPUT_DIR, "joy_direction.pt"))
save_direction(grief_direction, os.path.join(OUTPUT_DIR, "grief_direction.pt"))
save_direction(joy_direction_norm, os.path.join(OUTPUT_DIR, "joy_direction_norm.pt"))
save_direction(grief_direction_norm, os.path.join(OUTPUT_DIR, "grief_direction_norm.pt"))

# Save metadata
meta = {
    "model": MODEL_NAME,
    "steering_layer": steering_layer,
    "best_norm_layer": best_layer,
    "hidden_size": model.config.hidden_size,
    "n_layers": len(model.model.layers),
    "device": DEVICE,
    "dtype": str(DTYPE),
    "direction_norm": float(joy_direction.norm().item()),
    "layer_norms": norm_map,
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n\nSaved all outputs to {OUTPUT_DIR}/")
print(f"  - joy_direction.pt, grief_direction.pt")
print(f"  - results.json, outputs.txt")
print(f"  - metadata.json, pca_multi_layer.png")

print(f"\n{'='*70}")
print("EXPERIMENT 6 COMPLETE")
print(f"{'='*70}")
print(f"\nCompare these results with outputs/experiment_02_refined/ to see")
print(f"how the 1.5B model differs from the 0.5B model.")
