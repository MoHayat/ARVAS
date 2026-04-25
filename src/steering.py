"""
Activation steering utilities for injecting emotion direction vectors.

Supports both single-direction steering and 2D valence-arousal plane steering.
"""
import torch
from typing import Callable, Optional, Tuple
from baukit import TraceDict


def build_steering_hook(
    direction: torch.Tensor,
    alpha: float = 10.0,
) -> Callable:
    """Return an edit_output function that adds alpha * direction to activations.

    Args:
        direction: (hidden_dim,) steering vector.
        alpha: scaling coefficient.

    Returns:
        A function suitable for passing as edit_output to baukit.Trace/TraceDict.
    """
    def edit_output(output, layer_name):
        # output shape: (batch, seq_len, hidden_dim)
        # We add to ALL token positions for simplicity in generation
        steering = alpha * direction.to(device=output.device, dtype=output.dtype)
        # Unsqueeze to (1, 1, hidden_dim) for broadcasting
        return output + steering.unsqueeze(0).unsqueeze(0)

    return edit_output


def compute_2d_direction(
    valence_axis: torch.Tensor,
    arousal_axis: torch.Tensor,
    valence: float,
    arousal: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute a steering direction from 2D valence-arousal coordinates.

    Steering vector = valence * valence_axis + arousal * arousal_axis.

    Args:
        valence_axis: (hidden_dim,) principal component for valence.
        arousal_axis: (hidden_dim,) principal component for arousal.
        valence: scalar valence coordinate (-1 to +1).
        arousal: scalar arousal coordinate (-1 to +1).
        normalize: if True, L2-normalize the blended vector.

    Returns:
        direction: (hidden_dim,) steering vector.
    """
    direction = valence * valence_axis + arousal * arousal_axis
    if normalize:
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm
    return direction


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    layer_names: list,
    direction: torch.Tensor,
    alpha: float = 10.0,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt with activation steering applied.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        prompt: text prompt.
        layer_names: list of layer module paths to hook.
        direction: steering vector.
        alpha: steering strength.
        max_new_tokens: generation length.
        device: torch device.

    Returns:
        Generated text string.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = build_steering_hook(direction, alpha)

    with torch.no_grad():
        with TraceDict(model, layer_names, edit_output=hook) as _:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for comparison
            )

    # Decode only the new tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_with_2d_steering(
    model,
    tokenizer,
    prompt: str,
    layer_names: list,
    valence_axis: torch.Tensor,
    arousal_axis: torch.Tensor,
    valence: float,
    arousal: float,
    alpha: float = 10.0,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> str:
    """Generate text with 2D valence-arousal steering.

    Convenience wrapper that computes the blended direction from coordinates.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        prompt: text prompt.
        layer_names: list of layer module paths to hook.
        valence_axis: (hidden_dim,) valence principal component.
        arousal_axis: (hidden_dim,) arousal principal component.
        valence: scalar valence (-1 to +1).
        arousal: scalar arousal (-1 to +1).
        alpha: steering strength.
        max_new_tokens: generation length.
        device: torch device.

    Returns:
        Generated text string.
    """
    direction = compute_2d_direction(valence_axis, arousal_axis, valence, arousal)
    return generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer_names=layer_names,
        direction=direction,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
        device=device,
    )
