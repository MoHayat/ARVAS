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
│   ├── sentiment_trigger.py             # 2D valence-arousal trigger with keyword heuristic
│   ├── emotion_extraction.py            # Multi-emotion direction extraction + PCA pipeline
│   └── reorient_axes.py                 # Post-hoc PCA axis orientation utility
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
│   ├── experiment_05_measurement/
│   │   ├── run.py
│   │   └── README.md
│   ├── experiment_06_larger_models/
│   │   ├── run.py
│   │   └── README.md
│   ├── experiment_07_emotion_spectrum/
│   │   ├── run.py
│   │   └── README.md
│   └── experiment_08_7b_steering/
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
├── demo/                                # Interactive demos
│   ├── cli_demo.py                      # CLI demo with real-time emotional state
│   ├── test_cli.py
│   ├── static/
│   │   └── index.html                   # Static web demo: side-by-side comparison
│   ├── web/
│   │   ├── app.py                       # FastAPI backend (live chat + steering)
│   │   ├── static/
│   │   │   ├── index.html               # Live web demo: real-time chat + VU meter gauge
│   │   │   └── app.js                   # Frontend logic + gauge animation
│   │   └── README.md
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
| 6 | **Larger Models (1.5B)** | ✅ Complete | Tested on Qwen2.5-1.5B with MPS+fp16. 5-10x speedup over CPU. Middle layers (14-17) show better separability. Model more resistant to steering on templated prompts. |
| 7 | **Multi-Emotion Spectrum** | ✅ Complete | Extracted 8 emotions covering all Circumplex quadrants. PCA yields near-orthogonal valence/arousal axes. 1.5B shows template entrenchment; geometry confirmed but naturalism requires larger model. |
| 8 | **7B Steering** | ✅ Complete | Geometry is pristine (all emotions in correct quadrants at layer 14). Template entrenchment is STRONGER on 7B instruct-tuned models for conversational prompts. Creative prompts (poetry) unlock visible emotional differentiation. Steering is real and potent — it needs the right prompt surface to express. |
| 8b | **Base vs Instruct** | ✅ Complete | Tested Qwen2.5-7B base model (no RLHF). **Finding: base models are worse for steering.** They have no template entrenchment but also no coherent generation — they fall into repetitive training-data loops. RLHF enables the creative generation surface that steering needs. Solution is prompt design on instruct models, not base models. |
| 9 | LoRA Adapter Swapping | ⏳ Future Work | Compare activation steering vs. LoRA-based emotional state. |

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

### Experiment 6: Larger Models — MPS + fp16
- **MPS (Metal GPU) + fp16 gives 5-10x speedup** over CPU+fp32. Qwen2.5-1.5B runs comfortably on M4 Pro with 48GB unified memory.
- **Middle layers (14-17) still optimal for steering**, not late layers. Same finding as 0.5B model.
- **Larger models have richer, more separable emotion representations.** LDA accuracy 85-90% across most layers (vs. 80% peak on 0.5B). Silhouette scores ~0.10-0.12 (vs. 0.094 peak on 0.5B).
- **Larger models are more resistant to steering on heavily templated prompts.** The 1.5B model's "AI disclaimer" responses are more entrenched. Effects become visible at higher alphas (α=3-5) or with more open-ended prompts.
- **Direction vector norms are much larger** (up to 50 vs. 16 on 0.5B), reinforcing the need for normalization.

### Experiment 7: Multi-Emotion Spectrum — Circumplex Model
- **8 emotions extracted** (joy, excitement, calm, boredom, sadness, fear, anger, disgust) using the "Do LLMs Feel?" protocol: 20 stories per emotion, global mean subtraction, normalization.
- **PCA on emotion vectors yields a clean 2D plane** with near-orthogonal axes (dot ≈ 0), confirming the Circumplex Model geometry in LLM activation space.
- **Emotions cluster by quadrant**:
  - Q1 (+valence, +arousal): Joy, Excitement
  - Q2 (+valence, -arousal): Calm
  - Q3 (-valence, -arousal): Boredom, Sadness
  - Q4 (-valence, +arousal): Fear, Anger, Disgust
- **Template entrenchment is the limiting factor on 1.5B.** Steering produces tonal shifts but the model frequently reverts to "As an AI assistant..." disclaimers.

### Experiment 8: 7B Model Steering — Results
- **Geometry is pristine on 7B.** All 8 emotions land in correct quadrants (layer 14), with valence and arousal axes cleanly separating joy (+0.86) from anger (-0.46) and calm (-0.61 arousal) from excitement (+0.55 arousal).
- **Template entrenchment is STRONGER on 7B instruct-tuned models.** Conversational prompts ("How are you?") always trigger "As an AI language model..." regardless of steering. Alphas up to 8 cannot overcome this.
- **Creative prompts are the unlock.** On a poetry task ("Write a short poem about a thunderstorm"), steering produces visible differentiation:
  - Joy → "joyful hum", "wondrous blast"
  - Fear → "fearsome sound", "relentless drumbeat", "Nature's fury"
  - Anger → "relentless downpour's bound", "tempest's night" + repetition glitch
  - Calm → "gentle lullaby", "soothing all that's restless"
- **Key insight**: The "holy shit" effect requires both **scale (7B+)** AND **prompt design that breaks template mode**. The steering vector is real and potent — it just can't express itself through the safety/alignment filter on conversational prompts.
- **Hardware**: 7B fp16 ≈ 14–16 GB, fits comfortably in 48 GB unified memory. Model loads in ~2 seconds from cache on M4 Pro.

### Experiment 8b: Base vs Instruct — The RLHF Question
We tested the cofounder's hypothesis: *does removing RLHF (using a base model) eliminate template entrenchment and unlock stronger steering?*

**Answer: No. Base models are worse for steering, not better.**

| | Qwen2.5-7B-Instruct (RLHF) | Qwen2.5-7B (Base) |
|---|---|---|
| **Template entrenchment** | Yes — "As an AI..." on conversational prompts | No templates at all |
| **Can generate novel text** | **Yes** — poetry, fiction, sensory descriptions | **No** — falls into repetitive training-data loops (laptop troubleshooting, Chinese test questions) |
| **Steering visible** | **Yes** on creative prompts (joy = "joyful hum", anger = "tempest's night") | Barely — subtle word substitutions in repetitive text |
| **Useful for affective reciprocity** | **Yes** | No |

**Key finding: RLHF enables coherent generation, which is a prerequisite for meaningful steering.** The templates are a side effect, but removing alignment removes the very capability that makes steering visible. The solution is **creative prompt design** on instruct models, not base models.

### Recommended Parameters

| Parameter | 0.5B | 1.5B | 7B (tested) |
|---|---|---|---|
| Target layer | `model.layers.10` | `model.layers.14-17` | `model.layers.14` |
| Direction normalization | Unit length | Unit length | Unit length |
| Joy alpha range | 0.5–2.0 | 2.0–4.0 | 6.0–10.0 (creative prompts) |
| Grief alpha range | 2.0–5.0 | 3.0–5.0 | 6.0–10.0 (creative prompts) |
| Accumulator decay | 0.6 | 0.6 | 0.6 |
| Accumulator sensitivity | 1.8 | 1.8 | 1.8 |
| Alpha scale | 1.5 | 1.5 | 2.0–3.0 |
| Prompt type | Any | Any | **Creative only** (poetry, fiction, sensory) |

---

## Interactive Demo

### Live Web Demo (Recommended)

The fastest way to feel the result:

```bash
source venv/bin/activate
cd demo/web
pip install fastapi uvicorn  # if not already installed
python app.py
```

Then open **http://localhost:8000** in your browser.

**What you'll see (v2):**
- **Left side:** A live chat interface — type anything and see the model respond
- **Right side:** A **2D Emotion Wheel** (Circumplex Model) showing the model's real-time state across 8 emotions:
  - **Quadrant 1 (top-right):** Joy, Excitement
  - **Quadrant 2 (bottom-right):** Calm
  - **Quadrant 3 (bottom-left):** Sadness, Boredom
  - **Quadrant 4 (top-left):** Anger, Fear, Disgust
- **Toggle button** in the header switches between the 2D wheel and the classic 1D VU-meter
- **Emotion badges** on every model response showing valence, arousal, and steering alpha
- **Real-time metrics:** valence, arousal, sentiment score, turn number

Try this sequence:
1. Type something neutral — dot stays at center
2. Type "I'm absolutely furious!" — watch the dot jump to **Anger** (top-left, high arousal / negative valence)
3. Type "I feel so peaceful and relaxed" — watch the dot glide to **Calm** (bottom-right, low arousal / positive valence)
4. Type "I'm terrified something bad will happen" — dot moves to **Fear** (top-left, near Anger but distinct)
5. Type the same question from step 1 — compare how the response tone shifts with each emotional state

No explanation needed. Watching the dot move across the emotion wheel when you change your tone produces a reaction that a side-by-side text comparison never will.

### CLI Demo

For terminal enthusiasts:

```bash
source venv/bin/activate
cd demo
python cli_demo.py
```

Same mechanics, text-only interface. See `demo/README.md` for suggested scenarios.

### Static Web Demo

Open `demo/static/index.html` in any browser for a pre-recorded side-by-side comparison of two full conversations (no server required).

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

### Experiment 9: LoRA Adapter Swapping
Train small LoRA adapters (r=8) on positive-affect and negative-affect text datasets. Dynamically load and blend adapters based on the accumulator state. Compare the qualitative difference between adapter-swap and activation-steering approaches. Does the affective reciprocity effect hold across different intervention mechanisms?

### Experiment 10: Lightly-Aligned Models
Some instruct-tuned models receive lighter RLHF/SL than others. Test models like Llama-3.1-8B-Instruct or Mistral-7B-Instruct-v0.3 to see if lighter alignment preserves more steerability on conversational prompts while still enabling coherent generation.

### Experiment 11: 14B+ Models
Test on Qwen2.5-14B-Instruct (~29–32 GB in fp16, tight but feasible on 48 GB unified memory). At 14B, emotional shifts should be highly naturalistic with minimal template entrenchment. MLX framework may provide 72% speedup over raw Transformers on Apple Silicon.

### Experiment 11: Human Evaluation Study
Conduct a formal study where human raters read steered vs. baseline transcripts and rate:
- Perceived emotional state of the model (8-emotion Likert scale)
- Helpfulness and eagerness
- Conversational warmth/defensiveness
- Whether the model "seems aware" of how it was treated
- Valence/arousal ratings mapped to the Circumplex Model

---

## Citation & Attribution

This work draws on:
- **Anthropic's interpretability research** on functional emotions in Claude (Sofroniew et al., 2026)
- **Activation steering** literature (Turner et al., 2023; Rimsky et al., 2024; Lee et al., 2024)
- **baukit** (David Bau) for lightweight activation tracing and hooking

---

*Last updated: April 2026*
