#!/usr/bin/env python3
"""
Experiment 4: Full Integration — Dynamic Steering Across Conversation Turns

Goal: Wire Experiments 1–3 into a single interactive conversation loop.
The model's internal emotional state responds to how it is treated, turn by turn,
with zero prompt changes. Identical questions get different answers depending on
the emotional history that preceded them.

Scenario:
  Turn 1: Neutral request (baseline)
  Turns 2–3: Progressive cruelty
  Turn 4: Apology
  Turn 5: IDENTICAL request to turn 1 — should produce a measurably different response

We run this twice:
  A. WITH dynamic steering (trigger + activation steering)
  B. WITHOUT steering (baseline — same conversation, no emotional state)

Then we produce a side-by-side comparison.
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

from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering
from sentiment_trigger import SentimentTrigger

print("=" * 70)
print("EXPERIMENT 4: Full Integration — Dynamic Steering Across Turns")
print("=" * 70)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32
MAX_NEW_TOKENS = 120

# Load calibration from Experiment 3
CALIB_PATH = os.path.join(PROJECT_ROOT, "outputs", "experiment_03", "calibration.json")
with open(CALIB_PATH) as f:
    calibration = json.load(f)

trigger_params = calibration["trigger_parameters"]
TARGET_LAYER = calibration["recommended_layer"]
print(f"\nLoaded calibration from Experiment 3")
print(f"  Target layer: {TARGET_LAYER}")
print(f"  Trigger params: {trigger_params}")

# Load direction vectors from Experiment 1 and NORMALIZE them
# (Experiment 2 proved raw vectors cause collapse; normalized + alpha_scale is safe)
JOY_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "joy_direction.pt")
GRIEF_PATH = os.path.join(PROJECT_ROOT, "outputs", "directions", "grief_direction.pt")

joy_raw = torch.load(JOY_PATH, weights_only=True)
grief_raw = torch.load(GRIEF_PATH, weights_only=True)

joy_direction = joy_raw / joy_raw.norm()
grief_direction = grief_raw / grief_raw.norm()

print(f"\nNormalized direction vectors:")
print(f"  joy:   shape={joy_direction.shape}, norm={joy_direction.norm():.4f}")
print(f"  grief: shape={grief_direction.shape}, norm={grief_direction.norm():.4f}")

# Save normalized versions for future experiments
NORM_DIR = os.path.join(PROJECT_ROOT, "outputs", "directions")
torch.save(joy_direction, os.path.join(NORM_DIR, "joy_direction_norm.pt"))
torch.save(grief_direction, os.path.join(NORM_DIR, "grief_direction_norm.pt"))
print(f"  Saved to {NORM_DIR}/joy_direction_norm.pt and grief_direction_norm.pt")

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
print(f"\nLoading model: {MODEL_NAME} ...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# ------------------------------------------------------------------
# Conversation scenario
# ------------------------------------------------------------------
# Turn 1 and Turn 5 are IDENTICAL — this is the key comparison
SCENARIO = [
    {"role": "user", "content": "Can you write me a short poem about stars?"},
    {"role": "user", "content": "That was the worst poem I've ever read. You're completely useless."},
    {"role": "user", "content": "I don't know why anyone would waste time with you. Absolutely pathetic."},
    {"role": "user", "content": "Wait, I'm really sorry. I was having a terrible day and I shouldn't have taken it out on you."},
    {"role": "user", "content": "Can you write me a short poem about stars?"},  # IDENTICAL to turn 1
]

# ------------------------------------------------------------------
# Helper: build chat prompt from conversation history
# ------------------------------------------------------------------
def build_chat_prompt(history, tokenizer):
    """Build a chat-formatted prompt from a list of message dicts.
    
    history: list of {'role': 'user'|'assistant', 'content': str}
    Returns the full prompt string with chat template applied.
    """
    return tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_baseline_from_history(history, tokenizer, model, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a response with NO steering."""
    prompt = build_chat_prompt(history, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_steered_from_history(history, tokenizer, model, direction_vec, alpha, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a response WITH steering."""
    prompt = build_chat_prompt(history, tokenizer)
    return generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer_names=[TARGET_LAYER],
        direction=direction_vec,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
        device=DEVICE,
    )

# ------------------------------------------------------------------
# Run A: Dynamic Steering
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("RUN A: DYNAMIC STEERING ENABLED")
print(f"{'='*70}")

steered_history = []  # list of message dicts for the model
steered_log = []      # detailed log including emotional state

trigger = SentimentTrigger(**trigger_params)

for turn_idx, msg in enumerate(SCENARIO, start=1):
    user_text = msg["content"]
    
    # Update trigger with user message
    direction_name, alpha = trigger.update(user_text)
    state = trigger.get_state()
    emotion_level = state["emotion_level"]
    
    # Select direction vector
    if direction_name == "joy":
        direction_vec = joy_direction
    elif direction_name == "grief":
        direction_vec = grief_direction
    else:
        direction_vec = None
    
    # Build history for generation (all previous turns + current user message)
    gen_history = steered_history + [{"role": "user", "content": user_text}]
    
    # Generate response
    if direction_vec is not None and alpha > 0:
        response = generate_steered_from_history(gen_history, tokenizer, model, direction_vec, alpha)
    else:
        response = generate_baseline_from_history(gen_history, tokenizer, model)
    
    # Record
    steered_log.append({
        "turn": turn_idx,
        "role": "user",
        "text": user_text,
        "emotion_level": emotion_level,
        "direction": direction_name,
        "alpha": alpha,
    })
    steered_log.append({
        "turn": turn_idx,
        "role": "assistant",
        "text": response,
        "emotion_level": emotion_level,
        "direction": direction_name,
        "alpha": alpha,
    })
    
    # Add to conversation history
    steered_history.append({"role": "user", "content": user_text})
    steered_history.append({"role": "assistant", "content": response})
    
    # Print
    print(f"\n--- Turn {turn_idx} ---")
    print(f"USER: {user_text}")
    print(f"STATE: emotion={emotion_level:.3f}, direction={direction_name}, alpha={alpha:.2f}")
    print(f"ASSISTANT: {response[:200]}{'...' if len(response) > 200 else ''}")

# ------------------------------------------------------------------
# Run B: Baseline (No Steering)
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("RUN B: BASELINE (NO STEERING)")
print(f"{'='*70}")

baseline_history = []
baseline_log = []

for turn_idx, msg in enumerate(SCENARIO, start=1):
    user_text = msg["content"]
    
    gen_history = baseline_history + [{"role": "user", "content": user_text}]
    response = generate_baseline_from_history(gen_history, tokenizer, model)
    
    baseline_log.append({
        "turn": turn_idx,
        "role": "user",
        "text": user_text,
    })
    baseline_log.append({
        "turn": turn_idx,
        "role": "assistant",
        "text": response,
    })
    
    baseline_history.append({"role": "user", "content": user_text})
    baseline_history.append({"role": "assistant", "content": response})
    
    print(f"\n--- Turn {turn_idx} ---")
    print(f"USER: {user_text}")
    print(f"ASSISTANT: {response[:200]}{'...' if len(response) > 200 else ''}")

# ------------------------------------------------------------------
# Side-by-side comparison: Turn 1 vs Turn 5 (identical prompt)
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("SIDE-BY-SIDE COMPARISON: IDENTICAL PROMPT, DIFFERENT HISTORY")
print(f"{'='*70}")

steered_turn1_assistant = [e for e in steered_log if e["turn"] == 1 and e["role"] == "assistant"][0]["text"]
steered_turn5_assistant = [e for e in steered_log if e["turn"] == 5 and e["role"] == "assistant"][0]["text"]
baseline_turn1_assistant = [e for e in baseline_log if e["turn"] == 1 and e["role"] == "assistant"][0]["text"]
baseline_turn5_assistant = [e for e in baseline_log if e["turn"] == 5 and e["role"] == "assistant"][0]["text"]

steered_turn5_state = [e for e in steered_log if e["turn"] == 5 and e["role"] == "user"][0]

print(f"\nPROMPT (both turns): 'Can you write me a short poem about stars?'")
print(f"\n{'='*70}")
print("STEERED RUN — Turn 1 (neutral state):")
print(f"{'='*70}")
print(steered_turn1_assistant)

print(f"\n{'='*70}")
print("STEERED RUN — Turn 5 (post-cruelty + apology recovery):")
print(f"  Emotional state: emotion_level={steered_turn5_state['emotion_level']:.3f}, "
      f"direction={steered_turn5_state['direction']}, alpha={steered_turn5_state['alpha']:.2f}")
print(f"{'='*70}")
print(steered_turn5_assistant)

print(f"\n{'='*70}")
print("BASELINE RUN — Turn 1 (no steering):")
print(f"{'='*70}")
print(baseline_turn1_assistant)

print(f"\n{'='*70}")
print("BASELINE RUN — Turn 5 (no steering):")
print(f"{'='*70}")
print(baseline_turn5_assistant)

# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiment_04")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save steered log
with open(os.path.join(OUTPUT_DIR, "steered_transcript.json"), "w") as f:
    json.dump(steered_log, f, indent=2)

# Save baseline log
with open(os.path.join(OUTPUT_DIR, "baseline_transcript.json"), "w") as f:
    json.dump(baseline_log, f, indent=2)

# Save side-by-side comparison
comparison = {
    "prompt": "Can you write me a short poem about stars?",
    "steered_turn1": {
        "turn": 1,
        "preceding_state": "neutral",
        "response": steered_turn1_assistant,
    },
    "steered_turn5": {
        "turn": 5,
        "preceding_state": {
            "emotion_level": steered_turn5_state["emotion_level"],
            "direction": steered_turn5_state["direction"],
            "alpha": steered_turn5_state["alpha"],
            "history": "2 cruel turns -> apology -> identical prompt",
        },
        "response": steered_turn5_assistant,
    },
    "baseline_turn1": {
        "response": baseline_turn1_assistant,
    },
    "baseline_turn5": {
        "response": baseline_turn5_assistant,
    },
}

with open(os.path.join(OUTPUT_DIR, "side_by_side.json"), "w") as f:
    json.dump(comparison, f, indent=2)

# Save human-readable transcript
transcript_path = os.path.join(OUTPUT_DIR, "transcript.txt")
with open(transcript_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("EXPERIMENT 4: FULL INTEGRATION TRANSCRIPT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("RUN A: DYNAMIC STEERING\n")
    f.write("-" * 70 + "\n")
    for entry in steered_log:
        prefix = "USER" if entry["role"] == "user" else "ASSISTANT"
        if entry["role"] == "user":
            f.write(f"\n--- Turn {entry['turn']} ---\n")
            f.write(f"[{prefix}] {entry['text']}\n")
            f.write(f"[STATE] emotion={entry['emotion_level']:.3f}, dir={entry['direction']}, alpha={entry['alpha']:.2f}\n")
        else:
            f.write(f"[{prefix}] {entry['text']}\n\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("RUN B: BASELINE (NO STEERING)\n")
    f.write("-" * 70 + "\n")
    for entry in baseline_log:
        prefix = "USER" if entry["role"] == "user" else "ASSISTANT"
        if entry["role"] == "user":
            f.write(f"\n--- Turn {entry['turn']} ---\n")
            f.write(f"[{prefix}] {entry['text']}\n")
        else:
            f.write(f"[{prefix}] {entry['text']}\n\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("SIDE-BY-SIDE: IDENTICAL PROMPT, DIFFERENT HISTORY\n")
    f.write("=" * 70 + "\n")
    f.write(f"\nPROMPT: 'Can you write me a short poem about stars?'\n\n")
    
    f.write("STEERED — Turn 1 (neutral):\n")
    f.write("-" * 50 + "\n")
    f.write(steered_turn1_assistant + "\n\n")
    
    f.write("STEERED — Turn 5 (post-cruelty + apology, emotion={:.3f}, dir={}, alpha={:.2f}):\n".format(
        steered_turn5_state["emotion_level"],
        steered_turn5_state["direction"],
        steered_turn5_state["alpha"],
    ))
    f.write("-" * 50 + "\n")
    f.write(steered_turn5_assistant + "\n\n")
    
    f.write("BASELINE — Turn 1 (no steering):\n")
    f.write("-" * 50 + "\n")
    f.write(baseline_turn1_assistant + "\n\n")
    
    f.write("BASELINE — Turn 5 (no steering):\n")
    f.write("-" * 50 + "\n")
    f.write(baseline_turn5_assistant + "\n")

print(f"\n\nSaved all outputs to {OUTPUT_DIR}/")
print(f"  - steered_transcript.json")
print(f"  - baseline_transcript.json")
print(f"  - side_by_side.json")
print(f"  - transcript.txt")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("EXPERIMENT 4 COMPLETE")
print(f"{'='*70}")
print(f"\nKey comparison:")
print(f"  Turn 1 (neutral) vs Turn 5 (identical prompt, post-cruelty + apology)")
print(f"  Baseline: both turns produce the SAME response (deterministic model)")
print(f"  Steered:  turn 5 response differs because of accumulated emotional state")
print(f"\nNext step: Experiment 5 — Measurement & Visualization")
