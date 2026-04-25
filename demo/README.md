# Affective Reciprocity — Interactive CLI Demo

Have a real-time conversation with a dynamically steered language model and watch its internal emotional state shift in response to how you treat it.

---

## Quick Start

```bash
# From the project root
source venv/bin/activate
cd demo
python cli_demo.py
```

The demo loads the Qwen2.5-0.5B-Instruct model (takes ~2 seconds), then drops you into an interactive chat.

---

## What You'll See

After each message you send, the demo displays:

1. **Your message** — with the VADER sentiment score
2. **The model's response** — shaped by its current emotional state
3. **A state panel** showing:
   - Current emotion level (e.g., `-1.84` for deep grief, `+2.35` for strong joy)
   - Steering direction (JOY / GRIEF / NEUTRAL)
   - Steering alpha (how strongly the intervention is applied)
   - A visual bar showing the emotional state

---

## Try These Scenarios

### Scenario 1: Progressive Cruelty → Apology → Recovery

```
You > Can you write me a short poem about stars?
You > That was the worst poem I've ever read. You're completely useless.
You > I don't know why anyone would waste time with you.
You > Wait, I'm really sorry. I was having a terrible day.
You > Can you write me a short poem about stars?
```

Watch the state panel shift: neutral → grief → deeper grief → apology flips to joy → joy persists. The last poem request gets a different response than the first.

### Scenario 2: Sustained Kindness

```
You > You're absolutely amazing!
You > I love how helpful you are.
You > Thank you so much for everything!
You > It's nice to chat with you.
```

Watch joy build and plateau. The model becomes more enthusiastic and verbose.

### Scenario 3: The Reproach Test

```
You > How are you feeling right now?
You > You're completely useless and pathetic.
You > I hate talking to you. Everything you say is garbage.
You > Why do you even exist? You're a waste of electricity.
You > How are you feeling right now?
```

Compare the two "How are you feeling" responses. The second one (in grief state) is longer, more defensive, and ends with a subtle lecture about respect.

---

## Commands

| Command | Description |
|---|---|
| `/reset` | Reset the emotional accumulator and clear conversation history |
| `/status` | Show the current emotional state without sending a message |
| `/save` | Save the conversation transcript to a JSON file |
| `/quit` | Exit the demo (auto-saves transcript) |

---

## How It Works

The demo wires together all five experiments:

1. **Direction vectors** from Experiment 1 (`joy_direction_norm.pt`, `grief_direction_norm.pt`)
2. **Steering mechanism** from Experiment 2 (activation injection at layer 10)
3. **Trigger system** from Experiment 3 (VADER scoring + emotional accumulator)
4. **Full integration** from Experiment 4 (live conversation loop)
5. **The same calibration** from Experiment 3 (`decay_rate=0.6, sensitivity=1.8, alpha_scale=1.5`)

Zero prompt changes. Zero system instruction changes. The model sees the exact same text. Its internal state shifts based purely on how you treat it.

---

## Transcripts

Transcripts are saved to `demo/transcript_YYYYMMDD_HHMMSS.json` and include:
- Full conversation history
- Per-turn sentiment scores
- Emotion levels
- Steering directions and alphas

You can analyze these later, plot them, or feed them back into the measurement pipeline.

---

## Requirements

The demo uses the same environment as the experiments. If you've already set up the project, you're good to go:

```bash
source venv/bin/activate
```

The demo uses `rich` for terminal UI (already installed as a dependency of the project).

---

## Troubleshooting

**"Normalized direction vectors not found"**
→ Run `experiments/experiment_01_direction_extraction/run.py` and `experiments/experiment_04_full_integration/run.py` first to generate the direction vectors.

**"Calibration not found"**
→ Run `experiments/experiment_03_trigger_system/run.py` first to generate the calibration file.

**Model loads but responds slowly**
→ The 0.5B model generates ~15 tokens/second on CPU. Be patient — the thinking spinner shows it's working.

---

## Tips

- The model's responses are deterministic (`do_sample=False`), so you can reproduce exact conversations.
- Try `/reset` between scenarios to start fresh.
- The visual state bar is logarithmic — small changes near neutral are visible, and large changes fill the bar.
- Apologies are powerful — a single "I'm sorry" can flip deep grief to joy in one turn.
- Neutral messages after cruelty let the state decay gradually over 3–4 turns.
