"""
Activation steering utilities for injecting emotion direction vectors.
"""
import torch
from typing import Callable, Optional
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
    device = direction.device

    def edit_output(output, layer_name):
        # output shape: (batch, seq_len, hidden_dim)
        # We add to ALL token positions for simplicity in generation
        steering = alpha * direction.to(output.device)
        # Unsqueeze to (1, 1, hidden_dim) for broadcasting
        return output + steering.unsqueeze(0).unsqueeze(0)

    return edit_output


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
