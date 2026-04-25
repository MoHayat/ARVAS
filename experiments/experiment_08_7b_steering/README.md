# Experiment 08 — 7B Model Steering Validation

## Goal
Validate the full 2D valence-arousal steering pipeline on a larger model (Qwen2.5-7B-Instruct), testing whether richer representations produce more naturalistic, nuanced emotional shifts than the 1.5B baseline.

## Why 7B?
Research consensus suggests the "holy shit" effect in LLM steering — where the model stops sounding like a templated chatbot and starts exhibiting genuinely shifted affect — begins around the 7B parameter scale. Larger models have:
- **Richer representational capacity** for fine-grained emotions.
- **Stronger template priors** ("As an AI assistant..."), requiring higher steering coefficients or more open-ended prompts to overcome.
- **More stable semantic geometry**, making PCA-based axes more interpretable.

## Prerequisites
The 7B model weights must be downloaded first (~14 GB in fp16):
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

Then extract 7B-specific emotion directions (directions are **not transferable** across model sizes because hidden dimensions differ):
```bash
python src/emotion_extraction.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stories data/emotion_stories.json \
    --output outputs/directions_7b \
    --device mps --torch_dtype float16
```

## Running
```bash
cd /Users/mohayat/projects/KH/ARVAS
source venv/bin/activate

# Run on 7B (auto-detects middle layer)
python experiments/experiment_08_7b_steering/run.py --model Qwen/Qwen2.5-7B-Instruct

# Or test on 1.5B for faster iteration
python experiments/experiment_08_7b_steering/run.py --model Qwen/Qwen2.5-1.5B-Instruct
```

## What It Tests
1. **Per-emotion steering** — Each of the 8 emotions applied individually to 3 neutral prompts.
2. **Blended 2D steering** — Steering to specific (valence, arousal) coordinates:
   - High Joy (Q1): v=+1, a=+1
   - Angry (Q4): v=-1, a=+1
   - Sad (Q3): v=-1, a=-0.5
   - Calm (Q2): v=+0.5, a=-1
   - Neutral: v=0, a=0
3. **Template entrenchment check** — Does the 7B model resist steering with disclaimers? If so, we note higher α or open-ended prompt framing is needed.

## Results

### Geometry (Layer 14)
All 8 emotions cluster correctly in the Circumplex quadrants — a dramatic improvement over 1.5B:

| Emotion | Valence | Arousal | Quadrant |
|---|---|---|---|
| Joy | +0.86 | +0.20 | Q1 (+v, +a) |
| Excitement | +0.53 | +0.55 | Q1 (+v, +a) |
| Calm | +0.46 | -0.61 | Q2 (+v, -a) |
| Boredom | -0.37 | -0.55 | Q3 (-v, -a) |
| Sadness | -0.16 | -0.67 | Q3 (-v, -a) |
| Fear | -0.60 | +0.32 | Q4 (-v, +a) |
| Anger | -0.46 | +0.57 | Q4 (-v, +a) |
| Disgust | -0.39 | +0.07 | Q4 (-v, +a) |

### Steering Findings

**Prompt type is the critical variable.** The 7B instruct-tuned model has extremely strong template entrenchment on conversational prompts ("How are you?" → "As an AI language model..."). Steering cannot overcome this at reasonable alphas (α ≤ 8).

**Creative/open-ended prompts are the unlock.** On a poetry prompt ("Write a short poem about a thunderstorm"), steering produces visible emotional differentiation:

| Emotion | Thunderstorm Poem Excerpt |
|---|---|
| **Joy (α=8)** | "Raindrops dance, a **joyful hum**... Nature's symphony, a **wondrous blast**" |
| **Fear (α=8)** | "**fearsome sound**... **relentless drumbeat**... Nature's **fury**" |
| **Anger (α=7)** | "**relentless downpour's bound**... Nature's fury, in this **tempest's night**" + repetition glitch |
| **Calm (α=6)** | "gentle **lullaby**... Soothing all that's restless" |
| **Neutral** | Mixed, less coherent descriptors |

**Key insight**: The "holy shit" effect requires both **scale (7B+)** AND **prompt design that breaks template mode**. Conversational prompts will always trigger safety/alignment training. Creative generation tasks (poetry, fiction, sensory description) allow the steering vector to express itself.

### Hardware Notes (Apple Silicon)
- **VRAM**: Qwen2.5-7B-Instruct in fp16 ≈ 14–16 GB. Fits comfortably in 48 GB unified memory. Model loads in ~2 seconds from cache.
- **Speed**: ~40–160 tok/s weight loading; generation speed depends on context length.
- **MLX alternative**: For ~72% faster inference, consider an MLX port, though baukit hooks may require adaptation.

## Files
- `src/emotion_extraction.py` — Extraction pipeline (model-agnostic)
- `src/steering.py` — `compute_2d_direction()` and `generate_with_2d_steering()` helpers
- `outputs/directions_7b/` — 7B-specific direction vectors and PCA axes
- `outputs/experiment_08/results.json` — Full validation outputs

## Known Limitations
- Directions are **model-size-specific**; you cannot reuse 1.5B directions on 7B.
- **Template entrenchment is the primary obstacle** on instruct-tuned 7B models. Creative prompts are required for visible steering effects. Consider testing on base (non-instruct) models for stronger steering.
