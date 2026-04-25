"""
Multi-emotion direction extraction using the Circumplex Model.

Protocol (adapted from "Do LLMs Feel?" 2025 and Anthropic emotion-vector work):
1. For each emotion, collect short stories where a character experiences that emotion
   without naming it explicitly.
2. Extract residual-stream activations at middle layers for every story.
3. Compute per-emotion mean activation.
4. Subtract the global mean across ALL emotions to remove shared semantic structure.
5. Normalize each emotion direction to unit length.
6. Run PCA on the 8 emotion vectors; first two components = valence & arousal axes.
7. Save per-emotion directions + valence/arousal axes.

Usage:
    python src/emotion_extraction.py --model "Qwen/Qwen2.5-1.5B-Instruct" --stories data/emotion_stories.json --output outputs/directions
"""
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List

from activation_utils import load_model_and_tokenizer, get_layer_names, extract_activations


# Middle layers were optimal in Experiments 1-6; we use a band for robustness.
DEFAULT_LAYERS = ["model.layers.8", "model.layers.9", "model.layers.10", "model.layers.11"]

EMOTIONS = ["joy", "excitement", "calm", "boredom", "sadness", "fear", "anger", "disgust"]


def load_stories(path: str) -> Dict[str, List[str]]:
    with open(path) as f:
        data = json.load(f)
    # Ensure order matches EMOTIONS
    return {emo: data[emo] for emo in EMOTIONS}


def compute_emotion_directions(
    model,
    tokenizer,
    stories: Dict[str, List[str]],
    layer_names: List[str],
    device: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute mean activation per emotion, per layer.

    Returns:
        dict: {emotion: {layer_name: (hidden_dim,) tensor}}
    """
    directions = {}
    for emotion, texts in stories.items():
        print(f"Extracting activations for '{emotion}' ({len(texts)} stories)...")
        acts = extract_activations(
            model, tokenizer, texts, layer_names, device=device, last_token_only=True
        )
        # acts[layer] is (n_texts, hidden_dim); take mean across texts
        directions[emotion] = {layer: acts[layer].mean(dim=0) for layer in layer_names}
    return directions


def global_mean_center(
    emotion_directions: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Subtract global mean (across all emotions) per layer."""
    centered = {}
    layers = list(next(iter(emotion_directions.values())).keys())
    for layer in layers:
        # Stack all emotion means for this layer: (n_emotions, hidden_dim)
        stacked = torch.stack([emotion_directions[emo][layer] for emo in EMOTIONS], dim=0)
        global_mean = stacked.mean(dim=0)
        for emo in EMOTIONS:
            if emo not in centered:
                centered[emo] = {}
            centered[emo][layer] = emotion_directions[emo][layer] - global_mean
    return centered


def normalize_directions(
    directions: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """L2-normalize each direction vector."""
    normalized = {}
    for emo, layer_dict in directions.items():
        normalized[emo] = {}
        for layer, vec in layer_dict.items():
            norm = vec.norm()
            normalized[emo][layer] = vec / (norm if norm > 1e-8 else 1.0)
    return normalized


def pca_on_emotion_vectors(
    vectors: torch.Tensor,
    emotion_labels: List[str],
    n_components: int = 2,
) -> torch.Tensor:
    """Compute PCA components via SVD and orient them to align with expected valence/arousal.

    Orientation heuristic:
      - Valence (component 0) should point toward joy/excitement and away from sadness/anger.
      - Arousal (component 1) should point toward excitement/fear and away from calm/boredom.

    Args:
        vectors: (n_emotions, hidden_dim)
        emotion_labels: ordered list matching rows of `vectors`

    Returns:
        components: (n_components, hidden_dim)
    """
    # Center (upcast to float32 for SVD stability; SVD may not support fp16)
    X = vectors.float()
    mean = X.mean(dim=0, keepdim=True)
    X = X - mean
    # SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # Vh[0], Vh[1] are the top principal directions
    components = Vh[:n_components]

    # --- Orient valence axis (component 0) ---
    # Build a composite: joy + excitement - sadness - anger should have POSITIVE projection
    emo_idx = {e: i for i, e in enumerate(emotion_labels)}
    valence_ref = (
        vectors[emo_idx["joy"]]
        + vectors[emo_idx["excitement"]]
        - vectors[emo_idx["sadness"]]
        - vectors[emo_idx["anger"]]
    ).float()
    if torch.dot(components[0], valence_ref) < 0:
        components[0] = -components[0]

    # --- Orient arousal axis (component 1) ---
    # Build composite: excitement + fear - calm - boredom should have POSITIVE projection
    arousal_ref = (
        vectors[emo_idx["excitement"]]
        + vectors[emo_idx["fear"]]
        - vectors[emo_idx["calm"]]
        - vectors[emo_idx["boredom"]]
    ).float()
    if torch.dot(components[1], arousal_ref) < 0:
        components[1] = -components[1]

    return components


def extract_valence_arousal_axes(
    normalized_directions: Dict[str, Dict[str, torch.Tensor]],
    layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """For each layer, run PCA on the 8 emotion vectors and save top-2 components.

    Returns:
        {layer_name: (2, hidden_dim) tensor} where [0] = valence axis, [1] = arousal axis.
    """
    axes = {}
    for layer in layer_names:
        # Stack: (8, hidden_dim)
        stacked = torch.stack([normalized_directions[emo][layer] for emo in EMOTIONS], dim=0)
        components = pca_on_emotion_vectors(stacked, emotion_labels=EMOTIONS, n_components=2)
        axes[layer] = components
        print(f"  Layer {layer}: PCA component norms = "
              f"{components[0].norm().item():.3f}, {components[1].norm().item():.3f}")
    return axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--stories", default="data/emotion_stories.json")
    parser.add_argument("--output", default="outputs/directions")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--torch_dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.torch_dtype]

    # Load stories
    stories = load_stories(args.stories)
    print(f"Loaded {sum(len(v) for v in stories.values())} stories across {len(stories)} emotions.")

    # Load model
    print(f"Loading {args.model} on {args.device} with {args.torch_dtype}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device, torch_dtype=torch_dtype)
    model.eval()
    print(f"  Model loaded: {len(model.model.layers)} layers, {model.config.hidden_size} hidden dim")

    # Extract per-emotion mean activations
    emotion_directions = compute_emotion_directions(model, tokenizer, stories, args.layers, args.device)

    # Global mean centering
    print("\nSubtracting global mean across all emotions...")
    centered = global_mean_center(emotion_directions)

    # Normalize
    print("Normalizing direction vectors...")
    normalized = normalize_directions(centered)

    # Save per-emotion directions
    for emo in EMOTIONS:
        emo_path = out_dir / f"{emo}_direction.pt"
        # Save all layers as a dict; downstream code can pick one
        torch.save(normalized[emo], emo_path)
        print(f"  Saved {emo} direction -> {emo_path}")

    # Extract valence/arousal axes via PCA
    print("\nRunning PCA on emotion vectors to extract valence & arousal axes...")
    axes = extract_valence_arousal_axes(normalized, args.layers)

    for layer in args.layers:
        axis_path = out_dir / f"valence_arousal_axes_{layer.replace('.', '_')}.pt"
        torch.save(axes[layer], axis_path)
        print(f"  Saved valence/arousal axes for {layer} -> {axis_path}")

    print("\nDone! Direction extraction complete.")


if __name__ == "__main__":
    main()
