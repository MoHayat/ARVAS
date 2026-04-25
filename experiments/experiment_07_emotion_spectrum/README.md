# Experiment 07 — Multi-Emotion Spectrum Validation

## Goal
Validate that 8 emotion directions extracted via the Circumplex Model protocol produce distinct, recognizable emotional tones when steered individually, and that the 2D valence-arousal PCA plane correctly clusters emotions by quadrant.

## Method
1. **Dataset**: 160 short stories (20 per emotion) generated in `data/generate_emotion_stories.py`. Each vignette depicts a character experiencing an emotion without naming it explicitly.
2. **Extraction**: Run `src/emotion_extraction.py` on a model (1.5B or 7B) to:
   - Compute per-emotion mean activations at middle layers.
   - Subtract global mean across all emotions.
   - Normalize to unit vectors.
   - Run PCA → extract valence & arousal axes.
3. **Validation**: Load directions and steer a neutral prompt with each emotion vector. Record whether the model's tone shifts appropriately.

## Running
```bash
cd /Users/mohayat/projects/KH/ARVAS
source venv/bin/activate

# Extract directions (if not already done)
python src/emotion_extraction.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --stories data/emotion_stories.json \
    --output outputs/directions \
    --device mps --torch_dtype float16

# Run validation
python experiments/experiment_07_emotion_spectrum/run.py
```

## Key Findings
- **Geometry**: The first two PCA components on the 8 normalized emotion vectors are near-orthogonal (dot product ≈ 0), confirming a clean 2D plane.
- **Clustering** (1.5B model, layer 10):
  - **Positive valence**: Joy (+0.74), Calm (+0.81), Excitement (+0.04)
  - **Negative valence**: Anger (-0.58), Fear (-0.56), Disgust (-0.61), Boredom (-0.17)
  - **High arousal**: Joy (+0.45), Excitement (+0.73), Anger (+0.39)
  - **Low arousal**: Calm (-0.30), Boredom (-0.55), Sadness (-0.69)
- **Qualitative steering**: On the 1.5B model, template entrenchment is strong — the model often reverts to "As an AI assistant..." disclaimers even under steering. This validates the hypothesis that **larger models are needed** for naturalistic emotional expression.

## Files
- `data/generate_emotion_stories.py` — Story dataset generator
- `src/emotion_extraction.py` — Multi-emotion extraction + PCA pipeline
- `outputs/directions/*_direction.pt` — Per-emotion direction vectors
- `outputs/directions/valence_arousal_axes_*.pt` — 2D steering axes
- `outputs/experiment_07/results.json` — Validation outputs

## Next Step
Run Experiment 08 on the 7B model to test whether richer representations yield more naturalistic emotional steering.
