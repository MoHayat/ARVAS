#!/usr/bin/env python3
"""
Experiment 1 Runner: Extract joy/grief direction vectors from Qwen2.5-0.5B-Instruct.
This script mirrors the logic in notebooks/experiment_01_direction_extraction.ipynb.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import torch
import numpy as np
from tqdm import tqdm

from activation_utils import (
    load_model_and_tokenizer,
    get_layer_names,
    extract_activations,
    compute_mean_direction,
    save_direction,
)
from visualization import plot_activation_pca
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("=" * 60)
print("EXPERIMENT 1: Emotion Direction Extraction")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Load Model & Data
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32

print(f"\n[1/6] Loading model: {MODEL_NAME} ...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
print(f"  -> Layers: {len(model.model.layers)}, Hidden size: {model.config.hidden_size}")

# Load contrast pairs
with open("../data/contrast_pairs.json") as f:
    data = json.load(f)

positive_texts = data["positive"]
negative_texts = data["negative"]
print(f"\n[2/6] Loaded {len(positive_texts)} positive / {len(negative_texts)} negative samples")

# ------------------------------------------------------------------
# 2. Extract Activations at All Layers
# ------------------------------------------------------------------
layer_names = get_layer_names(model)
print(f"\n[3/6] Extracting activations across {len(layer_names)} layers ...")
print("      (This may take 2-3 minutes on CPU)")

pos_acts = extract_activations(model, tokenizer, positive_texts, layer_names, device=DEVICE)
neg_acts = extract_activations(model, tokenizer, negative_texts, layer_names, device=DEVICE)

sample_shape = pos_acts[layer_names[0]].shape
print(f"  -> Activations shape per layer: {sample_shape}  (n_texts, seq_len, hidden_dim)")

# ------------------------------------------------------------------
# 3. Compute Mean-Difference Direction Vectors Per Layer
# ------------------------------------------------------------------
print(f"\n[4/6] Computing direction vectors per layer ...")
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
# 4. Visualize with PCA (best layer)
# ------------------------------------------------------------------
print(f"\n[5/6] Generating PCA visualization for {best_layer} ...")
fig, pca_obj = plot_activation_pca(
    pos_acts[best_layer],
    neg_acts[best_layer],
    title=f"PCA of Last-Token Activations — {best_layer}",
    save_path="../outputs/figures/ex01_pca_best_layer.png",
)
print(f"  -> PCA explained variance: {pca_obj.explained_variance_ratio_}")

# Multi-layer PCA comparison
layers_to_viz = ["model.layers.5", "model.layers.10", "model.layers.15", "model.layers.20"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, layer_name in zip(axes, layers_to_viz):
    # pos_acts are now (n_texts, hidden_dim) since last_token_only=True
    pos = pos_acts[layer_name].numpy()
    neg = neg_acts[layer_name].numpy()
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

plt.suptitle("PCA Across Multiple Layers", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../outputs/figures/ex01_pca_multi_layer.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> Saved multi-layer PCA to ../outputs/figures/ex01_pca_multi_layer.png")

# ------------------------------------------------------------------
# 5. Save Final Direction Tensors
# ------------------------------------------------------------------
print(f"\n[6/6] Saving direction tensors ...")
joy_direction = directions[best_layer]
grief_direction = -joy_direction

save_direction(joy_direction, "../outputs/directions/joy_direction.pt")
save_direction(grief_direction, "../outputs/directions/grief_direction.pt")

print(f"  -> joy_direction.pt   (shape: {tuple(joy_direction.shape)}, norm: {joy_direction.norm():.4f})")
print(f"  -> grief_direction.pt (shape: {tuple(grief_direction.shape)}, norm: {grief_direction.norm():.4f})")

# Save metadata
meta = {
    "model": MODEL_NAME,
    "best_layer": best_layer,
    "hidden_size": model.config.hidden_size,
    "n_layers": len(model.model.layers),
    "n_positive_samples": len(positive_texts),
    "n_negative_samples": len(negative_texts),
    "direction_norm": float(joy_direction.norm().item()),
    "layer_norms": norm_map,
}

with open("../outputs/directions/ex01_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"  -> Saved metadata to ../outputs/directions/ex01_metadata.json")

# ------------------------------------------------------------------
# 6. Sanity Check
# ------------------------------------------------------------------
pos_centroid = pos_acts[best_layer].mean(dim=0)
neg_centroid = neg_acts[best_layer].mean(dim=0)
cos_sim = torch.nn.functional.cosine_similarity(pos_centroid, neg_centroid, dim=0).item()
euclidean = (pos_centroid - neg_centroid).norm().item()

print(f"\n[Sanity Check]")
print(f"  Cosine similarity (pos vs neg centroids): {cos_sim:.4f}")
print(f"  Euclidean distance:                       {euclidean:.4f}")
if cos_sim < 0.8:
    print("  -> Good separation! Directions are meaningfully different.")
else:
    print("  -> Centroids are quite similar. Consider more samples or a different layer.")

print("\n" + "=" * 60)
print("EXPERIMENT 1 COMPLETE")
print("=" * 60)
print(f"\nNext step: Experiment 2 — Static Steering Proof of Concept")
print(f"Target layer: {best_layer}")
