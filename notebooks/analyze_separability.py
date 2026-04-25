#!/usr/bin/env python3
"""
Quick analysis: compute separability metrics for key layers to validate
that the extracted directions are meaningful.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from activation_utils import load_model_and_tokenizer, extract_activations

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32

print("Loading model for separability analysis...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)

with open("../data/contrast_pairs.json") as f:
    data = json.load(f)

positive_texts = data["positive"]
negative_texts = data["negative"]

# Analyze a few key layers
test_layers = [
    "model.layers.5",   # early
    "model.layers.10",  # lower-middle
    "model.layers.15",  # upper-middle
    "model.layers.20",  # late
    "model.layers.23",  # last
]

print(f"\nAnalyzing {len(test_layers)} layers...\n")

for layer_name in test_layers:
    pos_acts = extract_activations(model, tokenizer, positive_texts, [layer_name], device=DEVICE, last_token_only=True)
    neg_acts = extract_activations(model, tokenizer, negative_texts, [layer_name], device=DEVICE, last_token_only=True)

    pos = pos_acts[layer_name].numpy()
    neg = neg_acts[layer_name].numpy()

    X = np.concatenate([pos, neg], axis=0)
    y = np.array([1] * len(pos) + [0] * len(neg))

    # LDA accuracy (leave-one-out is better with small N, but we'll do simple fit/predict)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    lda_pred = lda.predict(X)
    lda_acc = np.mean(lda_pred == y)

    # Silhouette score (higher = better separation)
    sil = silhouette_score(X, y)

    # Euclidean distance ratio
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)
    euclidean = np.linalg.norm(pos_centroid - neg_centroid)

    pos_std = np.std([np.linalg.norm(p - pos_centroid) for p in pos])
    neg_std = np.std([np.linalg.norm(n - neg_centroid) for n in neg])
    avg_std = (pos_std + neg_std) / 2
    ratio = euclidean / (avg_std + 1e-6)

    print(f"{layer_name}:")
    print(f"  LDA accuracy:      {lda_acc:.1%}")
    print(f"  Silhouette score:  {sil:.3f}")
    print(f"  Dist/Std ratio:    {ratio:.2f}")
    print(f"  Euclidean dist:     {euclidean:.2f}")
    print()

print("Done.")
