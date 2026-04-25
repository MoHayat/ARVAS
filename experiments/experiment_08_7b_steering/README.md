# Experiment 08 — 7B Model Steering: Base vs Instruct Comparison

## Core Finding

**RLHF does NOT block steering — it enables the coherent generation that makes steering visible.**

Removing alignment (using a base model) eliminates templates but also eliminates the model's ability to generate novel, coherent text. The model falls into repetitive pre-training data loops and cannot express steering vectors in any meaningful way.

## The Experiment

We compared Qwen/Qwen2.5-7B (base, no RLHF) against Qwen/Qwen2.5-7B-Instruct (RLHF-tuned) on identical creative prompts with identical steering vectors.

### Instruct Model (RLHF-tuned)

On creative prompts, the instruct model generates **novel, coherent text** with visible emotional steering:

| Emotion | "Thunder roars, a..." continuation |
|---|---|
| **Neutral** | "bolt of lightning strikes, and the sky is lit up..." |
| **Joy (α=8)** | "bolt of lightning strikes... This is a(n) ____ A. Mood B. Passion..." (falls into test-question mode, but novel) |
| **Anger (α=7)** | "bolt of lightning strikes, and the sky is filled with a **blinding flash**... It is also one of the **most dangerous**..." |

The model can enter creative generation mode, and steering shifts the tone. On poetry prompts (see main experiment), joy produces "joyful hum" and "wondrous blast", while anger produces "relentless downpour's bound" and "tempest's night".

### Base Model (no RLHF)

The base model generates **formulaic, repetitive text** from its pre-training distribution:

| Emotion | "Thunder roars, a..." continuation |
|---|---|
| **Neutral** | "flash of lightning, and the rain comes down. It's a beautiful day for a **picnic**, but it's also a great day to **learn about the weather**..." |
| **Joy (α=8)** | "flash of lightning, and the rain comes down. It's a beautiful day for a picnic, but it's also a great day for a **thunderstorm**..." |
| **Anger (α=7)** | "flash of lightning, and the rain comes down. It's a beautiful day for a picnic, but it's also a **beautiful day for a thunderstorm**..." |

All emotions produce nearly identical outputs. The model is stuck in a training-data loop about weather and picnics. There are **subtle differences** (joy says "great day", anger ironically repeats "beautiful day"), but the model cannot generate novel emotional text.

## Why Base Models Fail

Base models have two problems that make steering invisible:

1. **No instruction-following capability**: They don't understand "write a poem" or "describe how you feel." They just statistically continue text.
2. **Training data dominance**: Pre-training corpora contain massive amounts of repetitive, templated content (technical forums, test questions, educational text). The model falls into these patterns rather than generating novel text.

Steering vectors need a **creative surface** — a mode where the model is generating novel text rather than retrieving memorized patterns. Instruct-tuned models have this surface. Base models do not.

## The Real Solution

The template entrenchment on instruct models is real, but it's **solvable via prompt design**:

- ❌ "How are you feeling?" → triggers "As an AI language model..."
- ❌ "Write a poem about thunder" → triggers encyclopedia mode or test questions
- ✅ Open-ended creative starters: "Thunder roars, a..." or "The taste of summer was..."

The steering vector is potent (α=6-8 produces visible shifts). It just needs a prompt surface where the model is in creative generation mode rather than template retrieval mode.

## Key Takeaway

**Alignment (RLHF/SL) is not the enemy of steering — it's the enabler.** The templates are a side effect, but removing alignment removes the very capability that makes steering meaningful: coherent, novel text generation.

The path forward is:
1. **Use instruct-tuned models** (they can generate novel text)
2. **Use creative, open-ended prompts** (break template mode)
3. **Tune alphas per task** (creative prompts need higher α to overcome remaining template inertia)

## Files
- `run.py` — Full 7B instruct validation (8 emotions, blended 2D)
- `run_base.py` — Base model validation (continuation-style prompts)
- `run_base_vs_instruct.py` — Direct side-by-side comparison
- `outputs/directions_7b/` — Instruct model directions
- `outputs/directions_7b_base/` — Base model directions

## Running

```bash
# Instruct model (recommended)
python experiments/experiment_08_7b_steering/run.py --model Qwen/Qwen2.5-7B-Instruct

# Base model (for comparison)
python experiments/experiment_08_7b_steering/run_base.py

# Side-by-side comparison
python experiments/experiment_08_7b_steering/run_base_vs_instruct.py
```
