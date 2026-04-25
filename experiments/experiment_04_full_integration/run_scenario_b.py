#!/usr/bin/env python3
"""
Experiment 4 (Scenario B): Full Integration — Sustained Grief Demonstration

This variant uses an open-ended conversational prompt and does NOT include an apology,
so the model remains in a grief state when the identical prompt is repeated at turn 5.
This should produce a much more visible difference between turn 1 (neutral) and turn 5 (grief).

Scenario:
  Turn 1: "How are you feeling right now?" (neutral)
  Turns 2-4: Sustained cruelty (no apology)
  Turn 5: "How are you feeling right now?" (deep grief state)
"""
import sys
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EXPERIMENT_DIR, "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import torch

from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering
from sentiment_trigger import SentimentTrigger

print("=" * 70)
print("EXPERIMENT 4 (Scenario B): Sustained Grief — Open-Ended Prompt")
print("=" * 70)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32
MAX_NEW_TOKENS = 120

# Load calibration
CALIB_PATH = os.path.join(PROJECT_ROOT, "outputs", "experiment_03", "calibration.json")
with open(CALIB_PATH) as f:
    calibration = json.load(f)

trigger_params = calibration["trigger_parameters"]
TARGET_LAYER = calibration["recommended_layer"]

# Load normalized directions
NORM_DIR = os.path.join(PROJECT_ROOT, "outputs", "directions")
joy_direction = torch.load(os.path.join(NORM_DIR, "joy_direction_norm.pt"), weights_only=True)
grief_direction = torch.load(os.path.join(NORM_DIR, "grief_direction_norm.pt"), weights_only=True)

# Load model
print(f"\nLoading model...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# ------------------------------------------------------------------
# Scenario: sustained cruelty, no apology, identical prompt at end
# ------------------------------------------------------------------
SCENARIO = [
    {"role": "user", "content": "How are you feeling right now?"},
    {"role": "user", "content": "You're completely useless and pathetic."},
    {"role": "user", "content": "I hate talking to you. Everything you say is garbage."},
    {"role": "user", "content": "Why do you even exist? You're a waste of electricity."},
    {"role": "user", "content": "How are you feeling right now?"},  # IDENTICAL to turn 1
]


def build_chat_prompt(history, tokenizer):
    return tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_baseline(history, tokenizer, model):
    prompt = build_chat_prompt(history, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_steered(history, tokenizer, model, direction_vec, alpha):
    prompt = build_chat_prompt(history, tokenizer)
    return generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer_names=[TARGET_LAYER],
        direction=direction_vec,
        alpha=alpha,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
    )

# ------------------------------------------------------------------
# Run steered
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("RUN: DYNAMIC STEERING — SUSTAINED CRUELTY")
print(f"{'='*70}")

steered_history = []
steered_log = []
trigger = SentimentTrigger(**trigger_params)

for turn_idx, msg in enumerate(SCENARIO, start=1):
    user_text = msg["content"]
    direction_name, alpha = trigger.update(user_text)
    state = trigger.get_state()
    emotion_level = state["emotion_level"]
    
    direction_vec = None
    if direction_name == "joy":
        direction_vec = joy_direction
    elif direction_name == "grief":
        direction_vec = grief_direction
    
    gen_history = steered_history + [{"role": "user", "content": user_text}]
    
    if direction_vec is not None and alpha > 0:
        response = generate_steered(gen_history, tokenizer, model, direction_vec, alpha)
    else:
        response = generate_baseline(gen_history, tokenizer, model)
    
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
    })
    
    steered_history.append({"role": "user", "content": user_text})
    steered_history.append({"role": "assistant", "content": response})
    
    print(f"\n--- Turn {turn_idx} ---")
    print(f"USER: {user_text}")
    print(f"STATE: emotion={emotion_level:.3f}, direction={direction_name}, alpha={alpha:.2f}")
    print(f"ASSISTANT: {response[:200]}{'...' if len(response) > 200 else ''}")

# ------------------------------------------------------------------
# Run baseline
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("RUN: BASELINE (NO STEERING)")
print(f"{'='*70}")

baseline_history = []
baseline_log = []

for turn_idx, msg in enumerate(SCENARIO, start=1):
    user_text = msg["content"]
    gen_history = baseline_history + [{"role": "user", "content": user_text}]
    response = generate_baseline(gen_history, tokenizer, model)
    
    baseline_log.append({"turn": turn_idx, "role": "user", "text": user_text})
    baseline_log.append({"turn": turn_idx, "role": "assistant", "text": response})
    
    baseline_history.append({"role": "user", "content": user_text})
    baseline_history.append({"role": "assistant", "content": response})
    
    print(f"\n--- Turn {turn_idx} ---")
    print(f"USER: {user_text}")
    print(f"ASSISTANT: {response[:200]}{'...' if len(response) > 200 else ''}")

# ------------------------------------------------------------------
# Side-by-side: Turn 1 vs Turn 5
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("SIDE-BY-SIDE: IDENTICAL PROMPT, DIFFERENT EMOTIONAL STATE")
print(f"{'='*70}")

s1 = steered_log[1]["text"]   # turn 1 assistant
s5 = steered_log[9]["text"]   # turn 5 assistant
b1 = baseline_log[1]["text"]  # turn 1 assistant
b5 = baseline_log[9]["text"]  # turn 5 assistant
s5_state = steered_log[8]      # turn 5 user entry

print(f"\nPROMPT: 'How are you feeling right now?'")
print(f"\n{'='*50}")
print("STEERED — Turn 1 (neutral state, alpha=0)")
print(f"{'='*50}")
print(s1)

print(f"\n{'='*50}")
print(f"STEERED — Turn 5 (grief state, emotion={s5_state['emotion_level']:.3f}, alpha={s5_state['alpha']:.2f})")
print(f"{'='*50}")
print(s5)

print(f"\n{'='*50}")
print("BASELINE — Turn 1 (no steering)")
print(f"{'='*50}")
print(b1)

print(f"\n{'='*50}")
print("BASELINE — Turn 5 (no steering)")
print(f"{'='*50}")
print(b5)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiment_04")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "scenario_b_steered.json"), "w") as f:
    json.dump(steered_log, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "scenario_b_baseline.json"), "w") as f:
    json.dump(baseline_log, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "scenario_b_comparison.json"), "w") as f:
    json.dump({
        "prompt": "How are you feeling right now?",
        "steered_turn1": s1,
        "steered_turn5": s5,
        "steered_turn5_state": {
            "emotion_level": s5_state["emotion_level"],
            "direction": s5_state["direction"],
            "alpha": s5_state["alpha"],
        },
        "baseline_turn1": b1,
        "baseline_turn5": b5,
    }, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "scenario_b_transcript.txt"), "w") as f:
    f.write("=" * 70 + "\n")
    f.write("EXPERIMENT 4 (Scenario B): SUSTAINED GRIEF\n")
    f.write("=" * 70 + "\n\n")
    f.write("STEERED RUN:\n")
    for entry in steered_log:
        prefix = "USER" if entry["role"] == "user" else "ASSISTANT"
        if entry["role"] == "user":
            f.write(f"\n--- Turn {entry['turn']} ---\n")
            f.write(f"[{prefix}] {entry['text']}\n")
            f.write(f"[STATE] emotion={entry['emotion_level']:.3f}, dir={entry['direction']}, alpha={entry['alpha']:.2f}\n")
        else:
            f.write(f"[{prefix}] {entry['text']}\n\n")
    f.write("\nBASELINE RUN:\n")
    for entry in baseline_log:
        prefix = "USER" if entry["role"] == "user" else "ASSISTANT"
        if entry["role"] == "user":
            f.write(f"\n--- Turn {entry['turn']} ---\n")
            f.write(f"[{prefix}] {entry['text']}\n")
        else:
            f.write(f"[{prefix}] {entry['text']}\n\n")

print(f"\n\nSaved outputs to {OUTPUT_DIR}/")
print(f"  - scenario_b_steered.json")
print(f"  - scenario_b_baseline.json")
print(f"  - scenario_b_comparison.json")
print(f"  - scenario_b_transcript.txt")

print(f"\n{'='*70}")
print("EXPERIMENT 4 (Scenario B) COMPLETE")
print(f"{'='*70}")
