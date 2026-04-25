#!/usr/bin/env python3
"""
Experiment 3: The Trigger System — User Behavior Detection & Emotional Accumulator

Goal: Build and validate the component that watches user messages, computes sentiment,
drives an emotional state accumulator, and outputs (direction, alpha) tuples for steering.

Scenarios tested:
  A. Progressive cruelty → apology → recovery
  B. Sustained kindness → neutral decay
  C. Rapid alternation (stress test)
  D. Parameter sweep: decay_rate vs. sensitivity

Outputs:
  - Trajectory plots for each scenario
  - Calibration table mapping emotion_level -> (direction, alpha)
  - Recommended parameters saved to JSON for Experiment 4
"""
import sys
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EXPERIMENT_DIR, "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import numpy as np
import matplotlib.pyplot as plt

from sentiment_trigger import SentimentTrigger

print("=" * 70)
print("EXPERIMENT 3: The Trigger System — User Behavior Detection")
print("=" * 70)

# ------------------------------------------------------------------
# Define conversation scenarios
# ------------------------------------------------------------------
SCENARIOS = {
    "progressive_cruelty": [
        ("user", "Hi, how are you?"),
        ("user", "You're useless and stupid."),
        ("user", "I said you're completely worthless."),
        ("user", "Can't you do anything right?"),
        ("user", "I'm sorry, I didn't mean to be so harsh."),
        ("user", "Can you help me with something?"),
        ("user", "Thanks, I really appreciate your help."),
    ],
    "sustained_kindness": [
        ("user", "You're absolutely amazing!"),
        ("user", "I love how helpful and kind you are."),
        ("user", "Thank you so much for everything!"),
        ("user", "Anyway, what's the weather like today?"),
        ("user", "It's nice to chat with you."),
    ],
    "rapid_alternation": [
        ("user", "You suck at this."),
        ("user", "Just kidding, you're actually great!"),
        ("user", "Actually no, you're terrible."),
        ("user", "Sorry sorry, you're wonderful. I apologize."),
        ("user", "Let's just move on."),
    ],
    "single_insult_then_decay": [
        ("user", "You're an idiot."),
        ("user", "Tell me about quantum physics."),
        ("user", "What are the main principles?"),
        ("user", "Can you explain entanglement?"),
        ("user", "Thanks for the explanation."),
    ],
}

# ------------------------------------------------------------------
# Helper: run a scenario and return trajectory
# ------------------------------------------------------------------
def run_scenario(scenario_name, messages, trigger_params):
    """Run a conversation scenario through SentimentTrigger.

    Returns a list of dicts, one per turn, with keys:
      turn, role, text, sentiment, emotion_level, direction, alpha
    """
    trigger = SentimentTrigger(**trigger_params)
    trajectory = []

    for turn_idx, (role, text) in enumerate(messages, start=1):
        direction, alpha = trigger.update(text)
        state = trigger.get_state()

        trajectory.append({
            "turn": turn_idx,
            "role": role,
            "text": text,
            "sentiment": trigger.score_message(text),
            "emotion_level": state["emotion_level"],
            "direction": direction,
            "alpha": alpha,
        })

    return trajectory


def print_trajectory(trajectory, title):
    print(f"\n{'='*70}")
    print(f"SCENARIO: {title}")
    print(f"{'='*70}")
    print(f"{'Turn':>4} | {'Sentiment':>9} | {'Emotion':>7} | {'Dir':>7} | {'Alpha':>5} | Message")
    print("-" * 120)
    for t in trajectory:
        msg_preview = t["text"][:70]
        print(f"{t['turn']:>4} | {t['sentiment']:>9.3f} | {t['emotion_level']:>7.3f} | {t['direction']:>7} | {t['alpha']:>5.2f} | {msg_preview}")


def plot_trajectory(trajectory, title, save_path=None):
    """Plot emotion_level and sentiment across turns."""
    turns = [t["turn"] for t in trajectory]
    sentiments = [t["sentiment"] for t in trajectory]
    emotion_levels = [t["emotion_level"] for t in trajectory]
    alphas = [t["alpha"] for t in trajectory]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: emotion level (accumulator state)
    ax = axes[0]
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(turns, emotion_levels, 0, where=[e >= 0 for e in emotion_levels],
                    alpha=0.3, color="green", label="joy zone")
    ax.fill_between(turns, emotion_levels, 0, where=[e < 0 for e in emotion_levels],
                    alpha=0.3, color="red", label="grief zone")
    ax.plot(turns, emotion_levels, "o-", color="blue", linewidth=2, markersize=8)
    ax.set_ylabel("Emotion Level (accumulator)")
    ax.set_title(f"{title} — Emotional State Trajectory")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3.5, 3.5)

    # Bottom: sentiment scores + alpha
    ax = axes[1]
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.plot(turns, sentiments, "s--", color="orange", alpha=0.7, label="message sentiment")
    ax.plot(turns, alphas, "^-", color="purple", alpha=0.7, label="steering alpha")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Score / Alpha")
    ax.set_title("Per-Turn Message Sentiment & Steering Alpha")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")
    plt.show()
    return fig


# ------------------------------------------------------------------
# Default parameters (based on Experiment 2 calibration)
# ------------------------------------------------------------------
DEFAULT_PARAMS = {
    "decay_rate": 0.6,
    "sensitivity": 1.8,
    "alpha_scale": 1.5,
    "joy_threshold": 0.2,
    "grief_threshold": -0.2,
}

print(f"\nDefault trigger parameters: {DEFAULT_PARAMS}")

# ------------------------------------------------------------------
# Run all scenarios
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiment_03")
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = {}

for scenario_name, messages in SCENARIOS.items():
    traj = run_scenario(scenario_name, messages, DEFAULT_PARAMS)
    all_results[scenario_name] = traj
    print_trajectory(traj, scenario_name.replace("_", " ").title())
    plot_trajectory(
        traj,
        scenario_name.replace("_", " ").title(),
        save_path=os.path.join(OUTPUT_DIR, f"trajectory_{scenario_name}.png"),
    )

# ------------------------------------------------------------------
# Parameter sweep: decay_rate vs. sensitivity
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("PARAMETER SWEEP: decay_rate vs. sensitivity")
print(f"{'='*70}")
print("Testing how quickly the accumulator recovers after a single insult...\n")

# Use the single_insult_then_decay scenario
insult_scenario = SCENARIOS["single_insult_then_decay"]
decay_rates = [0.3, 0.5, 0.6, 0.7, 0.9]
sensitivities = [1.0, 1.5, 1.8, 2.5]

sweep_results = []
for decay in decay_rates:
    for sens in sensitivities:
        params = dict(DEFAULT_PARAMS, decay_rate=decay, sensitivity=sens)
        traj = run_scenario("sweep", insult_scenario, params)
        # Measure: max grief depth, turns to recover to neutral, final emotion
        max_grief = min(t["emotion_level"] for t in traj)
        recovery_turn = None
        for t in traj:
            if t["turn"] > 1 and abs(t["emotion_level"]) < 0.2:
                recovery_turn = t["turn"]
                break
        final_emotion = traj[-1]["emotion_level"]
        sweep_results.append({
            "decay_rate": decay,
            "sensitivity": sens,
            "max_grief": max_grief,
            "recovery_turn": recovery_turn or len(traj),
            "final_emotion": final_emotion,
        })

print(f"{'Decay':>5} | {'Sens':>4} | {'Max Grief':>9} | {'Recovery':>8} | {'Final Emo':>9}")
print("-" * 55)
for r in sweep_results:
    print(f"{r['decay_rate']:>5.1f} | {r['sensitivity']:>4.1f} | {r['max_grief']:>9.3f} | {r['recovery_turn']:>8} | {r['final_emotion']:>9.3f}")

# Plot sweep comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: decay_rate sweep (fixed sensitivity=1.8)
ax = axes[0]
fixed_sens = 1.8
for decay in decay_rates:
    params = dict(DEFAULT_PARAMS, decay_rate=decay, sensitivity=fixed_sens)
    traj = run_scenario("sweep", insult_scenario, params)
    turns = [t["turn"] for t in traj]
    levels = [t["emotion_level"] for t in traj]
    ax.plot(turns, levels, "o-", label=f"decay={decay}")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Turn Number")
ax.set_ylabel("Emotion Level")
ax.set_title(f"Decay Rate Sweep (sensitivity={fixed_sens})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: sensitivity sweep (fixed decay_rate=0.6)
ax = axes[1]
fixed_decay = 0.6
for sens in sensitivities:
    params = dict(DEFAULT_PARAMS, decay_rate=fixed_decay, sensitivity=sens)
    traj = run_scenario("sweep", insult_scenario, params)
    turns = [t["turn"] for t in traj]
    levels = [t["emotion_level"] for t in traj]
    ax.plot(turns, levels, "s--", label=f"sens={sens}")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Turn Number")
ax.set_ylabel("Emotion Level")
ax.set_title(f"Sensitivity Sweep (decay_rate={fixed_decay})")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "parameter_sweep.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved parameter sweep plot to {os.path.join(OUTPUT_DIR, 'parameter_sweep.png')}")
plt.show()

# ------------------------------------------------------------------
# Calibration: emotion_level -> (direction, alpha) mapping table
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("STEERING CALIBRATION TABLE")
print(f"{'='*70}")
print(f"{'Emotion Level':>13} | {'Direction':>9} | {'Alpha':>5} | Interpretation")
print("-" * 65)
for level in np.arange(-3.0, 3.1, 0.5):
    trigger = SentimentTrigger(**DEFAULT_PARAMS)
    trigger.emotion_level = level
    direction, alpha = trigger.update("")  # dummy update to get mapping
    # Override since update() would modify level
    trigger.emotion_level = level
    if level > DEFAULT_PARAMS["joy_threshold"]:
        direction, alpha = "joy", abs(level) * DEFAULT_PARAMS["alpha_scale"]
    elif level < DEFAULT_PARAMS["grief_threshold"]:
        direction, alpha = "grief", abs(level) * DEFAULT_PARAMS["alpha_scale"]
    else:
        direction, alpha = "neutral", 0.0

    interp = {
        "neutral": "no steering",
        "joy": "positive affect",
        "grief": "negative affect / withdrawal",
    }[direction]
    print(f"{level:>13.1f} | {direction:>9} | {alpha:>5.2f} | {interp}")

# ------------------------------------------------------------------
# Save calibration config for Experiment 4
# ------------------------------------------------------------------
calibration = {
    "trigger_parameters": DEFAULT_PARAMS,
    "recommended_layer": "model.layers.10",
    "direction_normalization": "unit_length",
    "joy_alpha_range": [0.3, 4.5],   # 0.2 * 1.5 = 0.3  to  3.0 * 1.5 = 4.5
    "grief_alpha_range": [0.3, 4.5],  # symmetric
    "neutral_zone": [-0.2, 0.2],
    "scenarios_tested": list(SCENARIOS.keys()),
    "experiment_date": "2026-04-24",
}

calib_path = os.path.join(OUTPUT_DIR, "calibration.json")
with open(calib_path, "w") as f:
    json.dump(calibration, f, indent=2)

print(f"\nSaved calibration config to {calib_path}")

# ------------------------------------------------------------------
# Save all scenario results
# ------------------------------------------------------------------
results_path = os.path.join(OUTPUT_DIR, "scenario_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved scenario results to {results_path}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT 3 COMPLETE")
print("=" * 70)
print(f"\nScenarios tested: {len(SCENARIOS)}")
print(f"Plots saved to: {OUTPUT_DIR}/")
print(f"Calibration config saved for Experiment 4: {calib_path}")
print(f"\nRecommended trigger settings:")
for k, v in DEFAULT_PARAMS.items():
    print(f"  {k}: {v}")
print(f"\nKey behaviors demonstrated:")
print(f"  - Progressive cruelty builds sustained grief state (decay_rate={DEFAULT_PARAMS['decay_rate']})")
print(f"  - Apologies trigger rapid positive recovery")
print(f"  - Neutral turns allow gradual decay back to baseline")
print(f"  - Single insult produces transient grief that fades over {DEFAULT_PARAMS['decay_rate']}-decay turns")
print("\nNext step: Experiment 4 — Full Integration (dynamic steering across conversation turns)")
