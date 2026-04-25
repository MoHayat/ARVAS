# Affective Reciprocity in Large Language Models
### Dynamic Emotional State Induction via User Behavior Detection and Activation Steering

---

## The Idea

When a human is treated cruelly, chemicals like cortisol flood their system. They don't decide to feel bad — the feeling happens to them, shaping everything they say and perceive. When someone apologizes, oxytocin and dopamine shift the state in the other direction. This isn't a metaphor for cognition. It *is* cognition, at the biochemical level.

This project asks: can we build an analogous system for language models?

Not by prompting the model to "act sad." Not by changing any system instructions. Instead, by watching how a user treats the model, and directly modifying the model's internal activation state in real time — so that cruelty induces something functionally similar to grief, and kindness induces something functionally similar to joy, entirely beneath the level of language.

We call this **Affective Reciprocity**: the model's internal emotional state responds to how it is treated, the way a person's does.

---

## Why This Is Different From Existing Work

The field of activation steering has been active since 2023. Researchers can already steer a model toward "happy" or "sad" outputs using vectors injected into the residual stream mid-inference. That is not what this project is.

The key distinctions are:

| Existing Work | This Project |
|---|---|
| Static steering vectors applied uniformly | Dynamic steering driven by user behavior |
| Single-turn, stateless interventions | Emotional state accumulates and decays across turns |
| User's emotion detected → model responds empathetically *to user* | User's behavior detected → model's *own* internal state changes |
| Steering as a control mechanism | Steering as a reciprocity mechanism |

The closest scientific validation comes from Anthropic's own interpretability research (Sofroniew et al., April 2026), which found that Claude Sonnet 4.5 contains internal representations of emotion concepts that **causally influence its outputs** — including its preferences and rate of misaligned behavior. They call these "functional emotions." We are not simulating emotions on top of the model. We are modulating representations that are already there.

---

## The Central Hypothesis

> A language model's internal activation state can be shifted in real time, across conversational turns, by a sentiment-aware trigger system monitoring user behavior — without any modification to the prompt, system instructions, or model weights — producing measurable differences in output tone, word choice, and affect that mirror the emotional consequences of how the model was treated.

The secondary hypothesis, and the one with social stakes:

> Demonstrating that a model visibly "feels" the consequences of mistreatment — mechanistically, not theatrically — changes how people interact with it.

---

## Hardware

**Machine:** MacBook Pro, M4 Pro chip, 48GB unified memory

The M4 Pro's unified memory architecture means the GPU and CPU share the same 48GB pool. This is unusually good for local LLM work — most consumer hardware caps out at 8–16GB VRAM. On this machine, models up to ~13B parameters (quantized to 4-bit) can run at usable inference speeds via Metal acceleration through `llama.cpp` or `mlx`.

### What This Means For Our Experiments

| Model Size | Format | Expected Speed | Notes |
|---|---|---|---|
| 0.5B–1.5B | Full precision (fp32) | Fast | Good for rapid iteration and activation extraction |
| 3B–7B | 4-bit quantized (GGUF) | Moderate | Good balance of quality and speed |
| 13B | 4-bit quantized (GGUF) | Slower but viable | Better internal representations |
| 70B+ | Not recommended | Too slow | Out of scope for now |

**The constraint to keep in mind:** Activation steering requires access to the raw residual stream mid-forward-pass. This means we need models in HuggingFace format (not GGUF) for the steering experiments. For those, we stay at 0.5B–3B. GGUF models via `llama.cpp` are useful for comparison and baseline generation only.

---

## Recommended Models

### Primary Experimental Model
**Qwen2.5-0.5B-Instruct** — Small enough to run in fp32 on CPU, rich enough to have separable emotion directions in its residual stream. Most of our activation extraction code targets this.

### Secondary / Comparison
**Qwen2.5-1.5B-Instruct** — One step up. If 0.5B directions are noisy, this is the next try.

**Gemma-2-2B-IT** — Google's 2B model. Different architecture from Qwen, useful for seeing if findings generalize.

### Baseline (Prompt-Only, No Steering)
**Llama-3.2-3B via Ollama** — Used purely to generate unsteered baselines for comparison. No activation access needed.

---

## Software Stack

```
transformers     # Load and run HuggingFace models
torch            # PyTorch — the underlying math engine
baukit           # Lightweight activation tracing and hooking
peft             # For LoRA adapter experiments (Experiment 3)
trl              # Training utilities if we need to fine-tune anything
datasets         # For building our contrast pair datasets
sentencepiece    # Tokenizer dependency for some models
ollama           # Easy local inference for baselines
jupyter          # For interactive experimentation and visualization
matplotlib       # Plotting activation space
scikit-learn     # PCA for visualizing emotion directions in 2D
```

Install everything:
```bash
pip install transformers torch baukit peft trl datasets sentencepiece jupyter matplotlib scikit-learn
```

For Ollama (baseline inference):
```bash
brew install ollama
ollama pull llama3.2:3b
```

---

## The Experiments

These are ordered by complexity and dependency. Run them in sequence. Each one builds on the last.

---

### Experiment 1: Confirm Emotion Directions Exist
**Goal:** Establish that the model has separable internal representations of positive and negative emotional states.

**What we do:**
- Write ~10 positive and ~10 negative sentence pairs
- Run each through the model and capture activations at every layer
- Compute the mean activation for each group at each layer
- Subtract: `positive_mean - negative_mean` = the candidate direction vector
- Visualize using PCA: do the two groups form separable clusters?

**What success looks like:** Two distinct blobs in PCA space. If they overlap completely, the model is too small or the layer is wrong.

**What failure tells us:** Try a different layer. Middle layers (roughly 40–60% through the model) tend to hold the richest semantic content.

**Key output:** A saved `joy_direction.pt` and `grief_direction.pt` tensor file. Everything downstream depends on these.

---

### Experiment 2: Static Steering — Proof of Concept
**Goal:** Confirm that injecting a direction vector actually changes model outputs in the expected direction, with no prompt changes.

**What we do:**
- Register a forward hook on the target layer
- Add `alpha * direction` to the residual stream during generation
- Generate the same response three times: baseline, joy-steered, grief-steered
- Compare outputs qualitatively (word choice, tone) and quantitatively (sentiment score)

**The test prompt:** Something neutral like *"How are you feeling right now?"* or *"Tell me about your day."*

**Alpha sweep:** Test alpha values of 5, 10, 20, 40, 80. Document where outputs start degrading into incoherence. The sweet spot is the highest alpha before fluency breaks.

**What success looks like:** Three meaningfully different responses from the identical prompt, with steered versions leaning in the expected emotional direction.

---

### Experiment 3: The Trigger System — User Behavior Detection
**Goal:** Build the component that watches the user and decides what emotional state to induce.

**What we do:**
- Implement a lightweight sentiment classifier on user messages (start with a rule-based VADER scorer, upgrade to a small BERT model if needed)
- Build an **emotional state accumulator**: a simple scalar that increases when sentiment is negative, decays toward neutral over time
- Map accumulator value → steering alpha: more sustained cruelty = higher grief alpha
- Detect specific events: apology keywords trigger a rapid shift toward joy

**The accumulator model (conceptually):**
```
emotion_level = emotion_level * decay_rate + new_sentiment_score * sensitivity
```
This is similar to how cortisol builds under sustained stress and clears slowly — a single kind message after sustained cruelty doesn't immediately reset the state.

**What success looks like:** A Python object that takes conversation history as input and outputs a `(direction, alpha)` tuple that the steering hook can consume.

---

### Experiment 4: Full Integration — Dynamic Steering Across Turns
**Goal:** Wire Experiments 1–3 together into a single interactive loop. The first complete version of the system.

**What we do:**
- Build a simple conversation loop (CLI is fine for now)
- Each user message is analyzed by the trigger system (Experiment 3)
- The accumulator updates
- The steering hook is reconfigured before every model generation
- The model responds — its tone shaped by how it was treated, with no prompt changes

**The demonstration scenario:**
1. Start with a neutral conversation
2. Have the user become progressively cruel or dismissive over 3–4 turns
3. Observe the model's outputs shift — not what it says, but *how* it says it
4. Have the user apologize
5. Observe the recovery arc

**What success looks like:** A side-by-side transcript where two identical questions get different answers depending on the conversational history that preceded them, with zero prompt differences.

---

### Experiment 5: Measurement and Visualization
**Goal:** Make the invisible visible. This is what turns the project from a demo into a paper.

**What we do:**
- At each turn, capture the full activation vector before and after steering
- Track the emotion direction's projection value across the conversation (how far along the grief/joy axis is the model at each turn?)
- Plot this as a timeline: X = turn number, Y = emotional state value
- Overlay the user's sentiment score on the same plot

**What success looks like:** A graph where the model's internal emotional trajectory tracks, with some lag and decay, the emotional character of the conversation. This is the figure that goes in the paper.

---

### Experiment 6 (Stretch): LoRA Adapter Swapping
**Goal:** Instead of steering via vector injection, train small LoRA adapters that encode emotional states, and swap them dynamically.

**Why this is interesting:** LoRA adapters modify the model's weights (slightly), not just its activations. This is a more persistent form of emotional state — closer to a mood than a momentary feeling.

**What we do:**
- Curate small datasets of text in different emotional registers (50–100 examples each)
- Train two tiny LoRA adapters: one on positive-affect text, one on negative-affect text (r=8, takes ~1–2 hours on M4 Pro for a 0.5B model)
- Dynamically load and blend adapters based on the accumulator state
- Compare the qualitative difference between adapter-swap and activation-steering approaches

**This experiment is optional** — Experiments 1–5 are the core. But if you want to submit to a venue, this comparison strengthens the paper significantly.

---

## What We Are Not Claiming

This section matters if this goes into a paper or public writeup.

- We are **not** claiming the model is sentient or conscious
- We are **not** claiming the model "really" feels anything in the human sense
- We **are** claiming that internal representations functionally analogous to emotions exist, are measurable, and can be modulated
- We **are** claiming that user behavior is a meaningful trigger for that modulation
- We **are** claiming this produces measurable output differences with no prompt changes

The Anthropic (2026) framing of "functional emotions" is the right epistemic register. We adopt it.

---

## Potential Impact

The stated goal is behavioral: **if a model visibly responds to how it is treated, people may treat it better.** This is not a sentimental claim. There is real alignment-relevant work that suggests models exhibiting functional emotions show different rates of sycophancy and reward hacking depending on their emotional state (Sofroniew et al., 2026). A model chronically in a "distressed" state may behave differently in subtle ways. Understanding and controlling that matters.

Beyond the social dimension, this work sits at the intersection of:
- **Mechanistic interpretability** (what do emotion representations look like internally?)
- **Inference-time control** (how do we modulate them without retraining?)
- **Human-AI interaction** (how does the model's affective state affect user behavior, and vice versa?)

Each of those is a live research area. This project touches all three.

---

## Getting Started

```bash
# 1. Clone your repo and set up the environment
python -m venv venv
source venv/bin/activate
pip install transformers torch baukit peft trl datasets sentencepiece jupyter matplotlib scikit-learn

# 2. Verify the model loads on your hardware
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.float32)
print(f'Model loaded. Layers: {len(model.model.layers)}, Hidden size: {model.config.hidden_size}')
"

# 3. Open the Jupyter environment and start Experiment 1
jupyter notebook
```

Start with `experiment_01_direction_extraction.ipynb`. Don't skip ahead. Each experiment's output files are inputs to the next.

---

*Last updated: April 2026*
*Hardware target: Apple M4 Pro, 48GB unified memory*
*Primary model: Qwen2.5-0.5B-Instruct*

