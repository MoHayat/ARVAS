"""
Re-orient valence/arousal axes after initial extraction.

The PCA components have arbitrary sign. This script orients them so that:
  - Valence axis points toward joy/excitement and away from sadness/anger.
  - Arousal axis points toward excitement/fear and away from calm/boredom.

Usage:
    python src/reorient_axes.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from emotion_extraction import pca_on_emotion_vectors, EMOTIONS

TARGET_LAYERS = [
    "model.layers.8",
    "model.layers.9",
    "model.layers.10",
    "model.layers.11",
]

def main():
    out_dir = PROJECT_ROOT / "outputs" / "directions"
    print("Re-orienting valence/arousal axes...")

    for layer in TARGET_LAYERS:
        # Load per-emotion directions
        vectors = []
        for emo in EMOTIONS:
            path = out_dir / f"{emo}_direction.pt"
            d = torch.load(path, weights_only=True)
            vectors.append(d[layer].float())
        stacked = torch.stack(vectors, dim=0)  # (8, hidden_dim)

        components = pca_on_emotion_vectors(stacked, emotion_labels=EMOTIONS, n_components=2)

        axis_path = out_dir / f"valence_arousal_axes_{layer.replace('.', '_')}.pt"
        torch.save(components, axis_path)
        print(f"  {layer}: valence norm={components[0].norm():.3f}, arousal norm={components[1].norm():.3f}")
        # Print projected coordinates for sanity check
        coords = {}
        for i, emo in enumerate(EMOTIONS):
            v = torch.dot(stacked[i], components[0]).item()
            a = torch.dot(stacked[i], components[1]).item()
            coords[emo] = (v, a)
            print(f"    {emo:12s} v={v:+.3f} a={a:+.3f}")

    print("Done. Axes re-oriented and saved.")


if __name__ == "__main__":
    main()
