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

## Expected Observations
- 7B responses should show **less rigid templating** than 1.5B under steering.
- Anger and fear steering may produce shorter, sharper responses.
- Joy/excitement may produce more elaborate, enthusiastic text.
- The 2D blended steering should produce **intermediate tones** (e.g., calm + slight joy = "warm contentment") that are impossible with a single binary joy/grief direction.

## Hardware Notes (Apple Silicon)
- **VRAM**: Qwen2.5-7B-Instruct in fp16 ≈ 14–16 GB. Fits comfortably in 48 GB unified memory.
- **Speed**: Estimated 13–40 tok/s on MPS depending on context length.
- **MLX alternative**: For ~72% faster inference, consider an MLX port, though baukit hooks may require adaptation.

## Files
- `src/emotion_extraction.py` — Extraction pipeline (model-agnostic)
- `src/steering.py` — `compute_2d_direction()` and `generate_with_2d_steering()` helpers
- `outputs/directions_7b/` — 7B-specific direction vectors and PCA axes
- `outputs/experiment_08/results.json` — Full validation outputs

## Known Limitations
- Directions are **model-size-specific**; you cannot reuse 1.5B directions on 7B.
- If the 7B model is too entrenched, you may need to increase `ALPHA_PER_EMOTION` constants in `run.py` or use more conversational prompts.
