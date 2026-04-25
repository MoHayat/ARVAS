# ARVAS: Affective Reciprocity in Large Language Models

**Dynamic Emotional State Induction via User Behavior Detection and Activation Steering**

---

## Overview

This project investigates whether a language model's internal activation state can be shifted in real time by monitoring user behavior — without changing prompts, system instructions, or model weights. We call this **Affective Reciprocity**: the model's internal "emotional" state responds to how it is treated, analogous to human emotional responses to social interaction.

The core mechanism is **activation steering**: extracting direction vectors in the model's residual stream that correspond to positive (joy) and negative (grief) affect, then injecting scaled versions of these vectors during inference based on a sentiment-aware trigger system.

---

## Hardware

- **Machine:** MacBook Pro, M4 Pro chip, 48GB unified memory
- **Primary model:** Qwen/Qwen2.5-0.5B-Instruct (24 layers, 896 hidden dim)
- **Software:** Python 3.13, PyTorch 2.11, Transformers, baukit

---

## Setup

```bash
# Create virtualenv with Python 3.13
python3.13 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
ARVAS/
├── overview.md                          # Full project context and theory
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── data/
│   └── contrast_pairs.json              # Positive/negative sentence pairs (Exp 1)
├── src/                                 # Reusable Python modules
│   ├── activation_utils.py              # Model loading, activation extraction, direction computation
│   ├── steering.py                        # Hook-based activation steering during generation
│   ├── sentiment_trigger.py             # VADER-based sentiment scoring and emotional accumulator
│   └── visualization.py                 # PCA plots, emotion timeline plots
├── experiments/                         # Self-contained experiment runners + READMEs
│   ├── experiment_01_direction_extraction/
│   │   ├── run.py
│   │   └── README.md
│   ├── experiment_02_static_steering/
│   │   ├── run.py
│   │   ├── run_refined.py
│   │   └── README.md
│   ├── experiment_03_trigger_system/
│   │   ├── run.py
│   │   └── README.md
│   ├── experiment_04_full_integration/
│   │   ├── run.py
│   │   ├── run_scenario_b.py
│   │   └── README.md
│   └── experiment_05_measurement/
│       ├── run.py
│       └── README.md
├── notebooks/                           # Jupyter notebook versions of experiments
│   ├── experiment_01_direction_extraction.ipynb
│   └── run_experiment_01.py
├── writeup/                             # Paper and blog post drafts
│   ├── paper.md
│   ├── blog_post.md
│   └── figures/
│       ├── fig1_pca_emotion_directions.png
│       ├── fig2_trigger_dynamics.png
│       ├── fig3_main_trajectory.png
│       └── fig4_intervention.png
├── demo/                                # Interactive CLI demo
│   ├── cli_demo.py
│   ├── test_cli.py
│   └── README.md
└── outputs/
    ├── directions/                      # joy_direction.pt, grief_direction.pt, *_norm.pt
    ├── figures/                         # PCA visualizations (Exp 1)
    ├── experiment_02/                   # Static steering results
    ├── experiment_02_refined/           # Refined sweet-spot results
    ├── experiment_03/                   # Trigger system calibration + trajectory plots
    ├── experiment_04/                   # Full integration transcripts
    └── experiment_05/                   # Measurement data + trajectory figures
```

---

## Experiments

| # | Name | Status | What Was Proved |
|---|---|---|---|
| 1 | **Direction Extraction** | ✅ Complete | Emotion directions exist in the residual stream and are separable by PCA. Middle layers (layer 10) encode the purest signal. |
| 2 | **Static Steering** | ✅ Complete | Activation steering changes model behavior. Layer 10 is optimal. Normalized directions with α=0.5–5 produce coherent but emotionally shifted outputs. Grief steering causes behavioral withdrawal. |
| 3 | **Trigger System** | ✅ Complete | Sentiment-aware emotional accumulator with realistic dynamics (buildup, decay, apology recovery). Fully calibrated to map emotion levels to safe steering alphas. |
| 4 | **Full Integration** | ✅ Complete | Live conversation loop where model responses shift based on accumulated emotional state. Identical prompts produce different answers depending on conversational history. Zero prompt changes. |
| 5 | **Measurement & Viz** | ✅ Complete | Internal activation trajectories captured and plotted. Natural state is blind to mistreatment; steering creates the emotional response. Publishable figures generated. |
| 6 | LoRA Adapter Swapping | ⏳ Future Work | Compare activation steering vs. LoRA-based emotional state. |
| 7 | Larger Models | ⏳ Future Work | Test on Qwen 1.5B, Gemma 2B, Llama 3B for stronger, more naturalistic shifts. |

---

## Key Findings

### Experiment 1: Direction Extraction
- **Best separability at layer 10** (not the last layer). Middle layers encode the richest semantic content.
- Positive and negative activations form **distinct PCA clusters**.
- `joy_direction.pt` and `grief_direction.pt` saved and ready for steering.

### Experiment 2: Static Steering
- **Raw direction vectors are too strong** for a 0.5B model — alphas ≥ 5 cause immediate output collapse.
- **Normalized directions + alphas 0.5–5** produce coherent, controllable shifts.
- **Layer 10 is the optimal steering point**.
- **Grief steering (α=5, layer 10)** caused refusal: *"I'm sorry, but I don't feel like talking about this."*
- **No prompt changes required.**

### Experiment 3: Trigger System
- **Apology detection creates rapid recovery** — a single apology flips grief to joy in one turn.
- **Accumulator self-regulates** — sustained kindness plateaus; sustained cruelty deepens but doesn't explode.
- **Calibrated mapping:** emotion_level × 1.5 = alpha, keeping all values within the coherent steering range.
- **Parameters:** `decay_rate=0.6, sensitivity=1.8` — ~3–4 turn mood duration.

### Experiment 4: Full Integration — The Core Result
- **Identical prompts produce different answers based on emotional history.**
- **Baseline proof:** A deterministic model without emotional state produces *identical* responses to identical prompts.
- **Steered proof:** The same prompt "How are you feeling right now?" produced:
  - *Turn 1 (joy α=0.35):* "...I'm **always ready to assist and provide information whenever you need help!** How can I assist you today?"
  - *Turn 5 (grief α=1.31):* "...It's important to remember that people should communicate with each other in a **respectful and understanding manner.**" — a subtle rebuke of earlier cruelty, something the baseline never does.
- **Grief steering made the model self-deprecating** ("Talking to me can be very frustrating and draining") and less eager to help.
- **Joy steering made responses more enthusiastic and personally engaged.**
- **Zero prompt or system instruction changes.** All differences come purely from activation vector injection.

### Experiment 5: Measurement & Visualization — The Paper Figure
- **The model's natural activation state is blind to mistreatment.** Even after three sustained insults, the natural projection at layer 10 remained **positive** (+1.4 to +1.5). The model reads its own helpfulness intent as positive, regardless of user cruelty.
- **Steering creates the emotional response.** The intervention pushes the effective state from +1.5 (oblivious) to -1.2 (grief) — a 2.7-point shift that makes the model "feel" the consequences of mistreatment.
- **The trajectory is publishable.** The plot shows: neutral → grief → deeper grief → sustained grief → partial recovery, overlayed with user sentiment and trigger state.
- **Epistemic reframing:** We are not "discovering hidden emotions." We are **engineering a functional emotion system** that induces state changes the model would not naturally produce.

### Recommended Parameters

| Parameter | Value |
|---|---|
| Target layer | `model.layers.10` |
| Direction normalization | Unit length |
| Joy alpha range | 0.5–2.0 |
| Grief alpha range | 2.0–5.0 |
| Accumulator decay | 0.6 |
| Accumulator sensitivity | 1.8 |
| Alpha scale | 1.5 |

---

## Interactive Demo

Want to experience it yourself? Run the CLI demo:

```bash
source venv/bin/activate
cd demo
python cli_demo.py
```

Chat with the model. Be kind. Be cruel. Apologize. Watch its internal emotional state update in real time after every message, and see how its responses shift accordingly.

See `demo/README.md` for suggested scenarios and commands.

---

## Write-Up

Two write-ups are available in `writeup/`:

- **`writeup/paper.md`** — Academic-style paper draft with full methods, results, discussion, and figures
- **`writeup/blog_post.md`** — Accessible narrative version for broader audiences

Key figures:
- `writeup/figures/fig1_pca_emotion_directions.png` — PCA clustering (Experiment 1)
- `writeup/figures/fig2_trigger_dynamics.png` — Trigger system dynamics (Experiment 3)
- `writeup/figures/fig3_main_trajectory.png` — The main paper figure (Experiment 5)
- `writeup/figures/fig4_intervention.png` — Steering intervention visualization (Experiment 5)

---

## Running an Experiment

Each experiment is **self-contained**. Just activate the virtualenv and run:

```bash
source venv/bin/activate
cd experiments/experiment_01_direction_extraction
python run.py
```

Results, figures, and a detailed README are generated automatically in that experiment's folder and in `outputs/`.

---

## Future Work

### Experiment 6: LoRA Adapter Swapping
Train small LoRA adapters (r=8) on positive-affect and negative-affect text datasets. Dynamically load and blend adapters based on the accumulator state. Compare the qualitative difference between adapter-swap and activation-steering approaches. Does the affective reciprocity effect hold across different intervention mechanisms?

### Experiment 7: Larger Models
Test on Qwen/Qwen2.5-1.5B-Instruct, Gemma-2-2B-IT, or Llama-3.2-3B via Ollama. Larger models may have:
- Stronger, more naturalistic emotional shifts
- Different optimal steering layers
- Less need for direction normalization (raw vectors may not cause collapse)
- More nuanced behavioral responses (subtlety rather than binary withdrawal/engagement)

### Human Evaluation Study
Conduct a formal study where human raters read steered vs. baseline transcripts and rate:
- Perceived emotional state of the model
- Helpfulness and eagerness
- Conversational warmth/defensiveness
- Whether the model "seems aware" of how it was treated

---

## Citation & Attribution

This work draws on:
- **Anthropic's interpretability research** on functional emotions in Claude (Sofroniew et al., 2026)
- **Activation steering** literature (Turner et al., 2023; Rimsky et al., 2024; Lee et al., 2024)
- **baukit** (David Bau) for lightweight activation tracing and hooking

---

*Last updated: April 2026*
