"""
Activation extraction and steering utilities for affective reciprocity experiments.
"""
import json
import torch
from typing import Dict, List, Tuple, Optional
from baukit import TraceDict
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cpu",
    torch_dtype=torch.float32,
):
    """Load a causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_layer_names(model) -> List[str]:
    """Return list of transformer block layer names for residual stream hooking.

    For Qwen2.5 models, layers are under model.model.layers[i].
    """
    names = []
    for i, layer in enumerate(model.model.layers):
        names.append(f"model.layers.{i}")
    return names


def extract_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_names: List[str],
    device: str = "cpu",
    last_token_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Extract residual-stream activations for a list of texts at given layers.

    Args:
        last_token_only: if True, returns only the last token's activation for each text.
                         This avoids variable-seq-length stacking issues and is what we
                         typically want for sentence-level direction extraction.

    Returns:
        If last_token_only=True: dict mapping layer_name -> (n_texts, hidden_dim)
        Otherwise: dict mapping layer_name -> list of (seq_len, hidden_dim) tensors
    """
    activations = {name: [] for name in layer_names}

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with TraceDict(model, layer_names, retain_output=True) as ret:
                _ = model(**inputs)
                for name in layer_names:
                    # ret[name].output is (batch, seq_len, hidden_dim)
                    out = ret[name].output.squeeze(0).cpu()  # (seq_len, hidden_dim)
                    if last_token_only:
                        activations[name].append(out[-1, :])  # (hidden_dim,)
                    else:
                        activations[name].append(out)

    if last_token_only:
        stacked = {}
        for name in layer_names:
            # List of (hidden_dim,) -> (n_texts, hidden_dim)
            stacked[name] = torch.stack(activations[name], dim=0)
        return stacked
    else:
        return activations


def compute_mean_direction(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    use_last_token: bool = True,
) -> torch.Tensor:
    """Compute mean-difference direction vector.

    Args:
        positive_acts: (n_pos, seq_len, hidden_dim) or (n_pos, hidden_dim)
        negative_acts: (n_neg, seq_len, hidden_dim) or (n_neg, hidden_dim)
        use_last_token: if True, use the last token position (typical for sentence-level meaning).
                        Only applies when inputs are 3D.

    Returns:
        direction: (hidden_dim,) vector = mean(pos) - mean(neg)
    """
    if positive_acts.dim() == 3 and use_last_token:
        pos = positive_acts[:, -1, :]  # (n_pos, hidden_dim)
        neg = negative_acts[:, -1, :]  # (n_neg, hidden_dim)
    else:
        pos = positive_acts
        neg = negative_acts

    direction = pos.mean(dim=0) - neg.mean(dim=0)
    return direction


def save_direction(direction: torch.Tensor, path: str):
    torch.save(direction, path)


def load_direction(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)


def flatten_activations_for_pca(activations_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dict of layer activations into (total_tokens, hidden_dim) for PCA.

    Each entry in activations_dict is (n_texts, seq_len, hidden_dim).
    We concatenate across all texts and all layers, but typically you'll call
    this on a single layer at a time.
    """
    # Assume single layer input for PCA visualization
    # (n_texts, seq_len, hidden_dim) -> (n_texts * seq_len, hidden_dim)
    tensor = list(activations_dict.values())[0]
    return tensor.view(-1, tensor.shape[-1])
