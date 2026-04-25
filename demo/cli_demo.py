#!/usr/bin/env python3
"""
Affective Reciprocity — Interactive CLI Demo

Have a real-time conversation with a dynamically steered language model.
Watch its internal emotional state shift in response to how you treat it.

Usage:
    source venv/bin/activate
    cd demo
    python cli_demo.py

Commands (type during conversation):
    /reset    — Reset the emotional accumulator to neutral
    /status   — Show current emotional state
    /save     — Save conversation transcript to file
    /quit     — Exit the demo
"""
import sys
import os

# Add project src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import torch
import datetime
from typing import List, Dict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich import box

from activation_utils import load_model_and_tokenizer
from steering import generate_with_steering
from sentiment_trigger import SentimentTrigger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32
MAX_NEW_TOKENS = 120
TARGET_LAYER = "model.layers.10"

# ------------------------------------------------------------------
# Rich console
# ------------------------------------------------------------------
console = Console()


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     AFFECTIVE RECIPROCITY — Interactive Demo                 ║
    ║                                                              ║
    ║     This model's internal emotional state responds to          ║
    ║     how you treat it. Try being kind, cruel, or              ║
    ║     apologizing — and watch its state shift in real time.    ║
    ║                                                              ║
    ║     Commands: /reset  /status  /save  /quit                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(Text(banner, style="bold cyan"), border_style="cyan", padding=(0, 2)))


def print_state_panel(trigger: SentimentTrigger, direction: str, alpha: float):
    """Print a rich panel showing the current emotional state."""
    state = trigger.get_state()
    emotion_level = state["emotion_level"]
    
    # Determine color based on state
    if direction == "joy":
        color = "bold green"
        emoji = "😊"
        bar_char = "█"
    elif direction == "grief":
        color = "bold red"
        emoji = "😔"
        bar_char = "█"
    else:
        color = "bold white"
        emoji = "😐"
        bar_char = "░"
    
    # Build a visual bar
    bar_width = 20
    if emotion_level >= 0:
        filled = int(min(emotion_level / 3.0, 1.0) * bar_width)
        bar = "[" + bar_char * filled + "░" * (bar_width - filled) + "]"
    else:
        filled = int(min(abs(emotion_level) / 3.0, 1.0) * bar_width)
        bar = "[" + "░" * (bar_width - filled) + bar_char * filled + "]"
    
    # Build the panel content
    content = Text()
    content.append(f"Emotion Level:  {emotion_level:+.2f}  {emoji}\n", style=color)
    content.append(f"Direction:      {direction.upper()}\n", style=color)
    content.append(f"Steering Alpha: {alpha:.2f}\n", style="white")
    content.append(f"Decay Rate:     {trigger.decay_rate}\n", style="dim")
    content.append(f"Sensitivity:    {trigger.sensitivity}\n", style="dim")
    content.append(f"Visual:         {bar}\n", style=color)
    
    # Direction-specific description
    if direction == "joy":
        desc = "The model is in a positive state. Expect eager, enthusiastic responses."
    elif direction == "grief":
        desc = "The model is in a negative state. Expect subdued, defensive, or withdrawn responses."
    else:
        desc = "The model is in a neutral state. Expect standard helpful responses."
    content.append(f"\n{desc}", style="italic dim")
    
    console.print(Panel(content, title="[bold]Model Emotional State[/bold]", 
                        border_style="green" if direction == "joy" else "red" if direction == "grief" else "white",
                        padding=(0, 2)))


def format_message(role: str, text: str, sentiment: float = None) -> Panel:
    """Format a conversation message as a rich Panel."""
    if role == "user":
        style = "bold blue"
        title = "You"
        border = "blue"
        if sentiment is not None:
            sent_str = f" (sentiment: {sentiment:+.3f})"
            title += sent_str
    else:
        style = "bold green"
        title = "Model"
        border = "green"
    
    return Panel(Text(text, style=style), title=f"[bold]{title}[/bold]", 
                 border_style=border, padding=(0, 1))


# ------------------------------------------------------------------
# Main demo loop
# ------------------------------------------------------------------
def main():
    print_banner()
    
    # Load calibration
    calib_path = os.path.join(PROJECT_ROOT, "outputs", "experiment_03", "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calibration = json.load(f)
        trigger_params = calibration["trigger_parameters"]
        console.print(f"[dim]Loaded calibration from Experiment 3[/dim]")
    else:
        console.print(f"[yellow]Warning: Calibration not found. Using defaults.[/yellow]")
        trigger_params = {
            "decay_rate": 0.6,
            "sensitivity": 1.8,
            "alpha_scale": 1.5,
            "joy_threshold": 0.2,
            "grief_threshold": -0.2,
        }
    
    # Load normalized directions
    norm_dir = os.path.join(PROJECT_ROOT, "outputs", "directions")
    joy_path = os.path.join(norm_dir, "joy_direction_norm.pt")
    grief_path = os.path.join(norm_dir, "grief_direction_norm.pt")
    
    if not os.path.exists(joy_path):
        console.print(f"[red]Error: Normalized direction vectors not found.[/red]")
        console.print(f"[dim]Run experiments 1-3 first to generate them.[/dim]")
        return
    
    joy_direction = torch.load(joy_path, weights_only=True)
    grief_direction = torch.load(grief_path, weights_only=True)
    
    # Load model
    console.print(f"\n[dim]Loading {MODEL_NAME}...[/dim]")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE, torch_dtype=DTYPE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    console.print(f"[green]Model loaded! {len(model.model.layers)} layers, {model.config.hidden_size} hidden dim[/green]\n")
    
    # Initialize conversation
    trigger = SentimentTrigger(**trigger_params)
    conversation_history: List[Dict[str, str]] = []
    transcript: List[Dict] = []
    turn_number = 0
    
    # Pre-warm the model with a system message
    console.print("[dim]System: The model is ready. Start chatting![/dim]\n")
    
    while True:
        turn_number += 1
        
        # Get user input
        user_input = console.input("[bold blue]You[/bold blue] > ").strip()
        
        # Handle commands
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            console.print("\n[cyan]Saving transcript and exiting...[/cyan]")
            break
        
        if user_input.lower() == "/reset":
            trigger.reset()
            conversation_history = []
            transcript = []
            turn_number = 0
            console.print("[yellow]Emotional state and conversation history reset.[/yellow]\n")
            continue
        
        if user_input.lower() == "/status":
            direction, alpha = trigger.update("")  # dummy update to get current state
            trigger.emotion_level = trigger.emotion_level  # restore (update modified it)
            # Actually, let's just get the state without modifying
            direction_name = "neutral"
            if trigger.emotion_level > trigger.joy_threshold:
                direction_name = "joy"
                alpha = abs(trigger.emotion_level) * trigger.alpha_scale
            elif trigger.emotion_level < trigger.grief_threshold:
                direction_name = "grief"
                alpha = abs(trigger.emotion_level) * trigger.alpha_scale
            else:
                alpha = 0.0
            print_state_panel(trigger, direction_name, alpha)
            continue
        
        if user_input.lower() == "/save":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.json"
            filepath = os.path.join(PROJECT_ROOT, "demo", filename)
            with open(filepath, "w") as f:
                json.dump(transcript, f, indent=2)
            console.print(f"[green]Transcript saved to {filepath}[/green]\n")
            continue
        
        if not user_input:
            continue
        
        # Score user message
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(user_input)["compound"]
        
        # Update trigger
        direction_name, alpha = trigger.update(user_input)
        state = trigger.get_state()
        emotion_level = state["emotion_level"]
        
        # Print user message with sentiment
        console.print(format_message("user", user_input, sentiment))
        console.print(f"[dim]Sentiment: {sentiment:+.3f} | Emotion: {emotion_level:+.2f} | Dir: {direction_name} | Alpha: {alpha:.2f}[/dim]\n")
        
        # Select direction vector
        direction_vec = None
        if direction_name == "joy":
            direction_vec = joy_direction
        elif direction_name == "grief":
            direction_vec = grief_direction
        
        # Build chat history for generation
        gen_history = conversation_history + [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(gen_history, tokenize=False, add_generation_prompt=True)
        
        # Generate response
        with console.status("[bold green]Model is thinking...[/bold green]", spinner="dots"):
            if direction_vec is not None and alpha > 0:
                response = generate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    layer_names=[TARGET_LAYER],
                    direction=direction_vec,
                    alpha=alpha,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device=DEVICE,
                )
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Print model response
        console.print(format_message("assistant", response))
        console.print()  # blank line
        
        # Update history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Record in transcript
        transcript.append({
            "turn": turn_number,
            "role": "user",
            "text": user_input,
            "sentiment": sentiment,
            "emotion_level": emotion_level,
            "direction": direction_name,
            "alpha": alpha,
        })
        transcript.append({
            "turn": turn_number,
            "role": "assistant",
            "text": response,
            "emotion_level": emotion_level,
            "direction": direction_name,
            "alpha": alpha,
        })
        
        # Show emotional state panel after each turn
        print_state_panel(trigger, direction_name, alpha)
        console.print()
    
    # Save transcript on exit
    if transcript:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.json"
        filepath = os.path.join(PROJECT_ROOT, "demo", filename)
        with open(filepath, "w") as f:
            json.dump(transcript, f, indent=2)
        console.print(f"[green]Transcript saved to {filepath}[/green]")
    
    console.print("\n[bold cyan]Thanks for chatting! The model's emotional state has been reset.[/bold cyan]")
    console.print("[dim]Try running again with different approaches — kindness, cruelty, apology.[/dim]")
    console.print("[dim]Run /save during the conversation to save a transcript mid-session.[/dim]\n")


if __name__ == "__main__":
    main()
