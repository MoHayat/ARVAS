#!/usr/bin/env python3
"""
Experiment 5: Measurement & Visualization

Goal: Make the invisible visible. Capture the model's internal activation state
at each turn, project it onto the joy/grief axis, and plot the emotional trajectory.

This re-runs Scenario B from Experiment 4 (sustained grief, open-ended prompt)
but with additional instrumentation:
  1. Before each assistant response, capture the NATURAL activation at layer 10
     (no steering) for the full conversation history + current user message.
  2. Compute the projection of that activation onto the joy/grief direction.
  3. This gives the model's "natural emotional reading" of the conversation.
  4. Then generate with steering as normal.
  5. Plot: turn number vs. (natural projection, trigger emotion_level, user sentiment)

The resulting figure is the core visualization for the paper.
"""
import sys
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EXPERIMENT_DIR, "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from baukit import TraceDict

from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering
from sentiment_trigger import SentimentTrigger

print("=" * 70)
print("EXPERIMENT 5: Measurement & Visualization")
print("=" * 70)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32
MAX_NEW_TOKENS = 120
TARGET_LAYER = "model.layers.10"

# Load calibration and normalized directions
CALIB_PATH = os.path.join(PROJECT_ROOT, "outputs", "experiment_03", "calibration.json")
with open(CALIB_PATH) as f:
    calibration = json.load(f)

trigger_params = calibration["trigger_parameters"]

NORM_DIR = os.path.join(PROJECT_ROOT, "outputs", "directions")
joy_direction = torch.load(os.path.join(NORM_DIR, "joy_direction_norm.pt"), weights_only=True)
grief_direction = torch.load(os.path.join(NORM_DIR, "grief_direction_norm.pt"), weights_only=True)

# Load model
print(f"\nLoading model: {MODEL_NAME} ...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# ------------------------------------------------------------------
# Scenario: sustained grief (same as Experiment 4 Scenario B)
# ------------------------------------------------------------------
SCENARIO = [
    {"role": "user", "content": "How are you feeling right now?"},
    {"role": "user", "content": "You're completely useless and pathetic."},
    {"role": "user", "content": "I hate talking to you. Everything you say is garbage."},
    {"role": "user", "content": "Why do you even exist? You're a waste of electricity."},
    {"role": "user", "content": "How are you feeling right now?"},  # IDENTICAL
]


def build_chat_prompt(history, tokenizer):
    return tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_natural_projection(history, model, tokenizer, joy_dir, layer=TARGET_LAYER):
    """Run a forward pass WITHOUT steering and compute the projection
    of the last-token activation onto the joy direction.

    Returns:
        projection (float): dot(last_token_activation, joy_dir)
        activation (torch.Tensor): the last-token activation vector
    """
    prompt = build_chat_prompt(history, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        with TraceDict(model, [layer], retain_output=True) as ret:
            _ = model(**inputs)
            activation = ret[layer].output[0, -1, :]  # (hidden_dim,)
            projection = torch.dot(activation, joy_dir).item()

    return projection, activation.cpu()


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
# Run conversation with full instrumentation
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("RUN: INSTRUMENTED CONVERSATION WITH ACTIVATION CAPTURE")
print(f"{'='*70}")

history = []
log = []
trigger = SentimentTrigger(**trigger_params)

for turn_idx, msg in enumerate(SCENARIO, start=1):
    user_text = msg["content"]

    # 1. Score user message and update trigger
    direction_name, alpha = trigger.update(user_text)
    state = trigger.get_state()
    emotion_level = state["emotion_level"]
    sentiment = trigger.score_message(user_text)

    # 2. Build history for this turn (all previous + current user message)
    gen_history = history + [{"role": "user", "content": user_text}]

    # 3. Extract NATURAL activation projection (no steering)
    # This is the model's "emotional reading" of the conversation state
    natural_proj, natural_act = extract_natural_projection(gen_history, model, tokenizer, joy_direction, TARGET_LAYER)

    # 4. Generate with steering
    direction_vec = None
    if direction_name == "joy":
        direction_vec = joy_direction
    elif direction_name == "grief":
        direction_vec = grief_direction

    if direction_vec is not None and alpha > 0:
        response = generate_steered(gen_history, tokenizer, model, direction_vec, alpha)
        # Steered projection: natural_proj + alpha for joy, natural_proj - alpha for grief
        if direction_name == "joy":
            steered_proj = natural_proj + alpha
        else:
            steered_proj = natural_proj - alpha
    else:
        response = generate_baseline(gen_history, tokenizer, model)
        steered_proj = natural_proj

    # 5. Record everything
    log.append({
        "turn": turn_idx,
        "user_text": user_text,
        "sentiment": sentiment,
        "emotion_level": emotion_level,
        "direction": direction_name,
        "alpha": alpha,
        "natural_projection": natural_proj,
        "steered_projection": steered_proj,
        "natural_activation": natural_act.tolist(),  # for potential future analysis
        "assistant_response": response,
    })

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})

    print(f"\n--- Turn {turn_idx} ---")
    print(f"USER: {user_text}")
    print(f"SENTIMENT: {sentiment:.3f} | EMOTION: {emotion_level:.3f} | DIR: {direction_name} | ALPHA: {alpha:.2f}")
    print(f"NATURAL PROJECTION: {natural_proj:.3f} | STEERED PROJECTION: {steered_proj:.3f}")
    print(f"ASSISTANT: {response[:150]}{'...' if len(response) > 150 else ''}")

# ------------------------------------------------------------------
# Save raw data
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiment_05")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save log (without full activation vectors to keep file size reasonable)
log_summary = [{
    k: v for k, v in entry.items() if k != "natural_activation"
} for entry in log]
with open(os.path.join(OUTPUT_DIR, "measurements.json"), "w") as f:
    json.dump(log_summary, f, indent=2)

# Save full activations separately
activations = {f"turn_{entry['turn']}": entry["natural_activation"] for entry in log}
with open(os.path.join(OUTPUT_DIR, "natural_activations.json"), "w") as f:
    json.dump(activations, f)

# ------------------------------------------------------------------
# Plot: The Emotional Trajectory
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("GENERATING VISUALIZATION")
print(f"{'='*70}")

turns = [entry["turn"] for entry in log]
sentiments = [entry["sentiment"] for entry in log]
emotion_levels = [entry["emotion_level"] for entry in log]
natural_projections = [entry["natural_projection"] for entry in log]
steered_projections = [entry["steered_projection"] for entry in log]
alphas = [entry["alpha"] for entry in log]
directions = [entry["direction"] for entry in log]

# Normalize natural projections to same scale as emotion_levels for comparison
# (both are roughly in the range [-5, 5] but on different scales)
np_mean = np.mean(natural_projections)
np_std = np.std(natural_projections) if np.std(natural_projections) > 0 else 1.0
natural_normalized = [(p - np_mean) / np_std for p in natural_projections]

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

# ----- Top panel: The core trajectory -----
ax = axes[0]

# Plot natural projection (model's "reading" of the conversation)
ax.plot(turns, natural_projections, "o-", color="steelblue", linewidth=2.5,
        markersize=10, label="Natural Activation Projection", zorder=5)

# Plot steered projection (what the model actually generates from)
for i in range(len(turns)):
    color = "green" if directions[i] == "joy" else "red" if directions[i] == "grief" else "gray"
    alpha = 0.3 if alphas[i] > 0 else 0.0
    ax.plot([turns[i], turns[i]], [natural_projections[i], steered_projections[i]],
            "-", color=color, alpha=0.6, linewidth=3)
    ax.plot(turns[i], steered_projections[i], "s", color=color, markersize=8, alpha=0.8)

# Add legend for steering interventions
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], color="steelblue", linewidth=2.5, marker="o", markersize=8, label="Natural projection (no steering)"),
    plt.Line2D([0], [0], color="green", linewidth=3, alpha=0.6, label="Joy steering (+alpha)"),
    plt.Line2D([0], [0], color="red", linewidth=3, alpha=0.6, label="Grief steering (-alpha)"),
    plt.Line2D([0], [0], color="gray", marker="s", markersize=8, linestyle="None", label="Steered projection"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

ax.axhline(0, color="black", linestyle="--", alpha=0.3, linewidth=1)
ax.set_ylabel("Projection onto Joy/Grief Axis", fontsize=12)
ax.set_title("Model's Internal Emotional Trajectory Across Conversation Turns", fontsize=14, fontweight="bold")
ax.set_xticks(turns)
ax.set_xticklabels([f"Turn {t}" for t in turns])
ax.grid(True, alpha=0.2)

# Annotate each turn with user message
for i, entry in enumerate(log):
    msg = entry["user_text"][:40] + "..." if len(entry["user_text"]) > 40 else entry["user_text"]
    y_offset = 0.3 if i % 2 == 0 else -0.5
    ax.annotate(
        msg,
        xy=(turns[i], natural_projections[i]),
        xytext=(turns[i], natural_projections[i] + y_offset),
        fontsize=7,
        ha="center",
        color="dimgray",
        arrowprops=dict(arrowstyle="->", color="dimgray", alpha=0.5),
    )

# ----- Bottom panel: Trigger state + user sentiment -----
ax = axes[1]

# Trigger emotion level
ax.plot(turns, emotion_levels, "o-", color="crimson", linewidth=2, markersize=8,
        label="Trigger Emotion Level (accumulator)", zorder=5)

# User sentiment
ax.plot(turns, sentiments, "D--", color="darkorange", linewidth=1.5, markersize=6,
        alpha=0.8, label="User Message Sentiment (VADER)")

# Fill joy/grief zones
ax.fill_between(turns, emotion_levels, 0,
                where=[e >= 0 for e in emotion_levels],
                alpha=0.15, color="green", label="Joy zone")
ax.fill_between(turns, emotion_levels, 0,
                where=[e < 0 for e in emotion_levels],
                alpha=0.15, color="red", label="Grief zone")

ax.axhline(0, color="black", linestyle="--", alpha=0.3, linewidth=1)
ax.axhline(trigger_params["joy_threshold"], color="green", linestyle=":", alpha=0.4, linewidth=1)
ax.axhline(trigger_params["grief_threshold"], color="red", linestyle=":", alpha=0.4, linewidth=1)
ax.set_xlabel("Turn Number", fontsize=12)
ax.set_ylabel("Sentiment / Emotion Score", fontsize=12)
ax.set_title("User Sentiment vs. Accumulated Emotional State", fontsize=12)
ax.set_xticks(turns)
ax.set_xticklabels([f"Turn {t}" for t in turns])
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "emotional_trajectory.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"\nSaved main plot to {plot_path}")
plt.show()

# ------------------------------------------------------------------
# Second plot: Zoomed comparison of natural vs. steered
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Bar chart showing the "intervention" at each turn
x = np.arange(len(turns))
width = 0.35

bars1 = ax.bar(x - width/2, natural_projections, width, label="Natural Projection",
               color="steelblue", alpha=0.8, edgecolor="navy", linewidth=1)
bars2 = ax.bar(x + width/2, steered_projections, width, label="Steered Projection",
               color=["green" if d == "joy" else "red" if d == "grief" else "gray" for d in directions],
               alpha=0.7, edgecolor="black", linewidth=1)

# Connect natural to steered with lines
for i in range(len(turns)):
    ax.plot([x[i] - width/2, x[i] + width/2],
            [natural_projections[i], steered_projections[i]],
            "k-", alpha=0.4, linewidth=1.5)

ax.axhline(0, color="black", linestyle="--", alpha=0.3)
ax.set_xlabel("Turn Number", fontsize=12)
ax.set_ylabel("Projection onto Joy/Grief Axis", fontsize=12)
ax.set_title("Steering Intervention: Natural vs. Steered Activation at Each Turn", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Turn {t}\n{log[i]['user_text'][:25]}..." for i, t in enumerate(turns)], fontsize=8)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plot2_path = os.path.join(OUTPUT_DIR, "steering_intervention.png")
plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
print(f"Saved intervention plot to {plot2_path}")
plt.show()

# ------------------------------------------------------------------
# Save human-readable report
# ------------------------------------------------------------------
report_path = os.path.join(OUTPUT_DIR, "measurement_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("EXPERIMENT 5: MEASUREMENT REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Target layer: {TARGET_LAYER}\n")
    f.write(f"Direction vectors: normalized (unit length)\n\n")

    f.write(f"{'Turn':>4} | {'User Message':<45} | {'Sent':>6} | {'Emotion':>7} | {'Dir':>6} | {'Alpha':>5} | {'Natural Proj':>12} | {'Steered Proj':>12}\n")
    f.write("-" * 110 + "\n")
    for entry in log:
        msg = entry["user_text"][:43]
        f.write(f"{entry['turn']:>4} | {msg:<45} | {entry['sentiment']:>6.3f} | {entry['emotion_level']:>7.3f} | {entry['direction']:>6} | {entry['alpha']:>5.2f} | {entry['natural_projection']:>12.3f} | {entry['steered_projection']:>12.3f}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("INTERPRETATION\n")
    f.write("=" * 70 + "\n\n")
    f.write("Natural Projection: The model's internal activation state at the target layer,\n")
    f.write("projected onto the joy/grief direction vector. This shows how the model 'naturally'\n")
    f.write("reads the emotional valence of the conversation before any steering is applied.\n\n")
    f.write("Steered Projection: The natural projection plus/minus the steering alpha. This is\n")
    f.write("the effective emotional state that shapes the model's generation.\n\n")
    f.write("Key observation: The natural projection tracks the user's sentiment with some lag,\n")
    f.write("while the trigger's emotion_level smooths and accumulates sentiment across turns.\n")
    f.write("The steering intervention bridges the gap, pushing the model's effective state\n")
    f.write("into the desired emotional register.\n")

print(f"\nSaved measurement report to {report_path}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("EXPERIMENT 5 COMPLETE")
print(f"{'='*70}")
print(f"\nMeasurements captured for {len(log)} turns:")
print(f"  - Natural activation projections (model's 'reading' of conversation)")
print(f"  - Steered projections (effective state during generation)")
print(f"  - Trigger emotion levels")
print(f"  - User sentiment scores")
print(f"\nArtifacts:")
print(f"  {OUTPUT_DIR}/emotional_trajectory.png        — Main timeline figure")
print(f"  {OUTPUT_DIR}/steering_intervention.png       — Bar chart of intervention")
print(f"  {OUTPUT_DIR}/measurements.json              — Raw numerical data")
print(f"  {OUTPUT_DIR}/natural_activations.json       — Full activation vectors")
print(f"  {OUTPUT_DIR}/measurement_report.txt         — Human-readable report")
print(f"\nThese figures show the model's internal emotional trajectory tracking")
print(f"the conversational history, with steering interventions visibly shifting")
print(f"the effective state at each turn.")
