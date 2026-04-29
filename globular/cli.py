#!/usr/bin/env python3
"""
Globular CLI - Full-featured Rich CLI
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

try:
    from huggingface_hub import HfApi, list_models, list_datasets, login, logout
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

VERSION = "0.1.0"
console = Console(force_terminal=False)


# =============================================================================
# AUTO-SCALING
# =============================================================================

def auto_scale(agents: int) -> dict:
    """
    Automatically scale parameters based on agent count.
    No caps - scales conservatively but unlimited.
    """
    agents = max(1, int(agents))
    if agents <= 32:
        agent_dim = 64
    else:
        # Power of 2 scaling
        import math
        agent_dim = 64 * (2 ** ((agents // 32) - 1).bit_length())
    
    # Steps: 2 + agents/64
    steps = max(2, 2 + (agents // 64))
    
    # Heads: must divide agent_dim evenly
    num_heads = 8
    while agent_dim % num_heads != 0 and num_heads > 1:
        num_heads //= 2
    
    return {
        "agent_dim": agent_dim,
        "steps": steps,
        "num_heads": num_heads,
    }


# =============================================================================
# DATA
# =============================================================================

MODELS = {
    "1": {"key": "qwen2-0.5b", "name": "Qwen2-0.5B", "hf": "Qwen/Qwen2-0.5B", "hidden": 896, "layers": 24},
    "2": {"key": "qwen2-1.5b", "name": "Qwen2-1.5B", "hf": "Qwen/Qwen2-1.5B", "hidden": 1536, "layers": 28},
    "3": {"key": "qwen2-7b", "name": "Qwen2-7B", "hf": "Qwen/Qwen2-7B", "hidden": 3584, "layers": 28},
    "4": {"key": "llama2-7b", "name": "Llama-2-7B", "hf": "meta-llama/Llama-2-7b", "hidden": 4096, "layers": 32},
    "5": {"key": "llama3-8b", "name": "Llama-3-8B", "hf": "meta-llama/Llama-3-8B", "hidden": 4096, "layers": 32},
}

PRESETS = {
    "1": {"key": "lambda", "name": "Lambda", "focus": "General", "agents": 32, "dim": 128, "steps": 3},
    "2": {"key": "opencode", "name": "Opencode", "focus": "Code", "agents": 32, "dim": 128, "steps": 3},
    "3": {"key": "opencode_reason", "name": "Opencode-Reason", "focus": "Code variant", "agents": 48, "dim": 192, "steps": 4},
}

SIZES = {
    "1": {"key": "light", "name": "Light", "agents": 16, "dim": 64, "steps": 2},
    "2": {"key": "standard", "name": "Standard", "agents": 32, "dim": 128, "steps": 3},
    "3": {"key": "heavy", "name": "Heavy", "agents": 64, "dim": 256, "steps": 4},
}


def calculate_params(agents: int) -> tuple:
    """Auto-calculate agent_dim and steps from agent count"""
    pow_val = agents // 32
    if pow_val <= 1:
        agent_dim = 64
    else:
        agent_dim = 64 * (2 ** (pow_val - 1).bit_length())
    steps = max(2, 2 + (agents // 64))
    return agent_dim, steps


def estimate(hidden, agents, agent_dim, steps):
    return agents * agent_dim * hidden * 2 + agents * agent_dim * steps * 4 + agents * agent_dim * agent_dim * 2


# =============================================================================
# MENUS
# =============================================================================

def header(title):
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan] [yellow]v{VERSION}[/yellow]", border_style="cyan", padding=(0, 2)))


def main_menu():
    console.clear()
    header("GLOBULAR CLI")
    console.print()
    console.print("[bold]Main Menu:[/bold]")
    console.print("  [1] Simple Mode - Quick preset-based")
    console.print("  [2] Expert Mode  - Full customization")
    console.print("  [3] Integrate  - Add to model")
    console.print("  [4] Train     - Train model")
    console.print("  [5] Publish   - Export model")
    console.print("  [6] HF Search  - Search HF")
    console.print()
    console.print("  [Q] Quit")
    
    return Prompt.ask("[cyan]>[/cyan]", choices=["1", "2", "3", "4", "5", "6", "q"], default="1")


# =============================================================================
# SIMPLE MODE
# =============================================================================

def simple_flow():
    """Simple preset-based flow"""
    console.clear()
    header("SIMPLE MODE")
    
    # Model
    console.print("\n[bold cyan]1.[/bold cyan] Model")
    for k, m in MODELS.items():
        console.print(f"    [{k}] {m['name']}")
    console.print("    [6] Custom")
    model_choice = Prompt.ask("[cyan]>[/cyan]", choices=list(MODELS.keys()) + ["6"], default="1")
    
    if model_choice == "6":
        model_name = Prompt.ask("[cyan]Model path[/cyan]", default="Qwen/Qwen2-0.5B")
    else:
        model_name = MODELS[model_choice]["hf"]
    
    # Preset
    console.print("\n[bold cyan]2.[/bold cyan] Preset")
    for k, p in PRESETS.items():
        console.print(f"    [{k}] {p['name']} - {p['focus']}")
    preset_choice = Prompt.ask("[cyan]>[/cyan]", choices=list(PRESETS.keys()), default="1")
    preset = PRESETS[preset_choice]
    
    # Size
    console.print("\n[bold cyan]3.[/bold cyan] Size")
    for k, s in SIZES.items():
        console.print(f"    [{k}] {s['name']} ({s['agents']} agents)")
    console.print("    [4] Heavy+ (custom - auto-scales)")
    size_choice = Prompt.ask("[cyan]>[/cyan]", choices=list(SIZES.keys()) + ["4"], default="2")
    
    if size_choice == "4":
        # Heavy+ with custom agents and auto-scaling
        custom_agents = Prompt.ask("[cyan]Agent count (or 'auto' for 128)[/cyan]", default="128")
        if custom_agents.lower() == "auto":
            custom_agents = "128"
        try:
            agents = int(custom_agents)
        except ValueError:
            agents = 128
        auto = auto_scale(agents)
        console.print(f"[dim]Auto-scaled: {agents} agents -> dim={auto['agent_dim']}, steps={auto['steps']}, heads={auto['num_heads']}[/dim]")
        size = {"name": "Heavy+", "agents": agents, "dim": auto["agent_dim"], "steps": auto["steps"]}
    else:
        size = SIZES[size_choice]
    
    # Layer
    console.print("\n[bold cyan]4.[/bold cyan] Insert after")
    console.print("    [1] Layer 5")
    console.print("    [2] Layer 10")
    console.print("    [3] Layers 5, 10, 15")
    layer_choice = Prompt.ask("[cyan]>[/cyan]", choices=["1", "2", "3"], default="1")
    layers_map = {"1": [5], "2": [10], "3": [5, 10, 15]}
    layers = layers_map[layer_choice]
    
    # Calculate
    hidden = MODELS.get(model_choice, {"hidden": 896}).get("hidden", 896)
    params = estimate(hidden, size["agents"], size["dim"], size["steps"])
    total = params * len(layers)
    
    # Summary
    summary = Table(box=None, pad_edge=True)
    summary.add_column("", style="cyan")
    summary.add_column("", style="white")
    summary.add_row("Model", model_name)
    summary.add_row("Preset", preset["name"])
    summary.add_row("Size", size["name"])
    summary.add_row("Layers", str(layers))
    summary.add_row("", "")
    summary.add_row("[green]Per Layer", f"[green]{params:,}[/green]")
    summary.add_row("[green]Total", f"[green]{total:,}[/green]")
    
    console.print()
    console.print(Panel(summary, title="[bold green]Summary[/bold green]", border_style="green"))
    console.print()
    
    Prompt.ask("[dim]Press Enter...[/dim]", default="")


# =============================================================================
# EXPERT MODE
# =============================================================================

def expert_flow():
    """Full customization flow"""
    console.clear()
    header("EXPERT MODE")
    
    console.print("\n[bold cyan]1.[/bold cyan] Model Config")
    model_name = Prompt.ask("[cyan]Model HF path[/cyan]", default="Qwen/Qwen2-0.5B")
    dim = Prompt.ask("[cyan]Hidden dimension[/cyan]", default="896")
    layers = Prompt.ask("[cyan]Number of layers[/cyan]", default="24")
    heads = Prompt.ask("[cyan]Attention heads[/cyan]", default="14")
    
    console.print("\n[bold cyan]2.[/bold cyan] Globular Config")
    console.print("    [auto] Auto-scale based on agents")
    agents_input = Prompt.ask("[cyan]Number of agents[/cyan]", default="32")
    
    # Handle auto-scaling
    if agents_input.lower() == "auto":
        agents = 128
    else:
        agents = int(agents_input)
    
    # Auto-detect other parameters
    console.print("\n  [auto] Auto-scale other params")
    console.print("  [custom] Enter manually")
    auto_choice = Prompt.ask("[cyan]>[/cyan]", choices=["auto", "custom"], default="auto")
    
    if auto_choice == "auto":
        auto = auto_scale(agents)
        agent_dim = auto["agent_dim"]
        steps = auto["steps"]
        num_heads = auto["num_heads"]
        console.print(f"[dim]Auto-scaled: dim={agent_dim}, steps={steps}, heads={num_heads}[/dim]")
    else:
        agent_dim = Prompt.ask("[cyan]Agent dimension[/cyan]", default="128")
        steps = Prompt.ask("[cyan]Evolution steps[/cyan]", default="3")
        num_heads = Prompt.ask("[cyan]Agent heads[/cyan]", default="8")
    
    console.print("\n[bold cyan]3.[/bold cyan] Layer Insertion")
    console.print("    [1] Specific layers")
    console.print("    [2] Every N layers")
    console.print("    [3] All layers")
    insert_choice = Prompt.ask("[cyan]>[/cyan]", choices=["1", "2", "3"], default="1")
    
    if insert_choice == "1":
        insert_layers = [int(x) for x in Prompt.ask("[cyan]Layers (e.g., 5,10,15)[/cyan]", default="5,10").split(",")]
    elif insert_choice == "2":
        step = Prompt.ask("[cyan]Insert every N layers[/cyan]", default="5")
        insert_layers = [int(step)]
    else:
        insert_layers = list(range(int(layers)))
    
    console.print("\n[bold cyan]4.[/bold cyan] Advanced Options")
    console.print("    Replace FFN:    [1] Yes  [2] No")
    replace_ffn = Prompt.ask("[cyan]>[/cyan]", choices=["1", "2"], default="2") == "1"
    console.print("    Enhance attention: [1] Yes  [2] No")
    enhance_attn = Prompt.ask("[cyan]>[/cyan]", choices=["1", "2"], default="1") == "1"
    
    console.print("\n[bold cyan]5.[/bold cyan] Training (optional)")
    do_train = Confirm.ask("[green]Train model?[/green]", default=False)
    
    if do_train:
        train_epochs = Prompt.ask("[cyan]Epochs[/cyan]", default="3")
        train_batch = Prompt.ask("[cyan]Batch size[/cyan]", default="4")
        train_lr = Prompt.ask("[cyan]Learning rate[/cyan]", default="1e-4")
        train_dataset = Prompt.ask("[cyan]Dataset[/cyan]", default="synthetic")
    
    # Validate inputs
    try:
        int(dim), int(layers), int(heads)
        int(agents), int(agent_dim), int(steps), int(num_heads)
    except ValueError:
        console.print("[red]Invalid numeric input[/red]")
        Prompt.ask("[dim]Press Enter...[/dim]", default="")
        return
    
    # Summary
    console.print()
    summary = Table(box=None, pad_edge=True)
    summary.add_column("", style="cyan")
    summary.add_column("", style="white")
    summary.add_row("Model", model_name)
    summary.add_row("Hidden Dim", dim)
    summary.add_row("Layers", str(layers))
    summary.add_row("", "")
    summary.add_row("Agents", str(agents))
    summary.add_row("Agent Dim", str(agent_dim))
    summary.add_row("Steps", str(steps))
    summary.add_row("Agent Heads", str(num_heads))
    summary.add_row("", "")
    summary.add_row("Insert After", str(insert_layers))
    summary.add_row("Replace FFN", str(replace_ffn))
    summary.add_row("Enhance Attn", str(enhance_attn))
    
    # Calculate params
    p = estimate(int(dim), int(agents), int(agent_dim), int(steps))
    total = p * len(insert_layers)
    summary.add_row("", "")
    summary.add_row("[green]Per Layer", f"[green]{p:,}[/green]")
    summary.add_row("[green]Total", f"[green]{total:,}[/green]")
    
    console.print(Panel(summary, title="[bold green]Config[/bold green]", border_style="green"))
    console.print()
    
    # Actions
    console.print("[bold]Actions:[/bold]")
    console.print("  [1] Save config (JSON)")
    console.print("  [2] Integrate + Save")
    if do_train:
        console.print("  [3] Train + Integrate")
    console.print("  [4] Back")
    
    action_choices = ["1", "2", "3", "4"] if do_train else ["1", "2", "4"]
    action = Prompt.ask("[cyan]>[/cyan]", choices=action_choices, default="4")
    
    if action == "1":
        # Save config
        config = {
            "model": model_name,
            "dim": int(dim),
            "layers": int(layers),
            "heads": int(heads),
            "agents": int(agents),
            "agent_dim": int(agent_dim),
            "steps": int(steps),
            "num_heads": int(num_heads),
            "insert_layers": insert_layers,
            "replace_ffn": replace_ffn,
            "enhance_attention": enhance_attn,
        }
        os.makedirs("./output", exist_ok=True)
        
        save_name = Prompt.ask("[cyan]Config name[/cyan]", default="globular_config")
        save_path = f"./output/{save_name}.json"
        
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]Saved to {save_path}[/green]")
        
        # Also save as main config
        with open("./output/config.json", "w") as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]Saved to ./output/config.json[/green]")
        
    elif action == "2":
        console.print(f"[dim]Integrating {model_name}...[/dim]")
        try:
            import torch
            from globular import GlobularIntegrationConfig, apply_globular_to_model
            from globular.hf_utils import safe_load_causal_lm, safe_load_tokenizer, ensure_tokenizer_padding
            
            model = safe_load_causal_lm(model_name, device_map="cpu", torch_dtype=torch.float32)
            tokenizer = ensure_tokenizer_padding(safe_load_tokenizer(model_name))
            
            config = GlobularIntegrationConfig(
                insert_after_layers=insert_layers,
                num_agents=int(agents),
                agent_dim=int(agent_dim),
                steps=int(steps),
                model_dim=int(dim),
                num_heads=int(heads),
                replace_ffn=replace_ffn,
                enhance_attention=enhance_attn,
            )
            
            model = apply_globular_to_model(model, config)
            
            output_dir = Prompt.ask("[cyan]Output dir[/cyan]", default="./output/globular-integrated")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save config alongside
            with open(f"{output_dir}/config.json", "w") as f:
                json.dump({"model": model_name, "agents": int(agents), "layers": insert_layers}, f)
            
            console.print(f"[green]Saved to {output_dir}[/green]")
            console.print(f"[dim]To generate: python -m globular.generate {output_dir}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
    
    elif action == "3" and do_train:
        console.print("[yellow]Training + integration:[/yellow]")
        console.print(f"  1. Run: python -m globular.train --agents {agents} --epochs {train_epochs}")
        console.print(f"  2. Then use [2] above with trained checkpoint")
    
    Prompt.ask("[dim]Press Enter...[/dim]", default="")


# =============================================================================
# INTEGRATE
# =============================================================================

def integrate_flow():
    console.clear()
    header("INTEGRATE")
    
    console.print("\n[bold cyan]1.[/bold cyan] Select model to integrate into")
    for k, m in MODELS.items():
        console.print(f"    [{k}] {m['name']}")
    console.print("    [6] Custom HF model")
    model_choice = Prompt.ask("[cyan]Model[/cyan]", choices=list(MODELS.keys()) + ["6"], default="1")
    
    if model_choice == "6":
        hf_model = Prompt.ask("[cyan]HuggingFace model path[/cyan]", default="Qwen/Qwen2-0.5B")
    else:
        hf_model = MODELS[model_choice]["hf"]
    
    console.print("\n[bold cyan]2.[/bold cyan] Globular config")
    console.print("    [1] Light (16 agents)")
    console.print("    [2] Standard (32 agents)")
    console.print("    [3] Heavy (64 agents)")
    console.print("    [4] Heavy+ (custom - auto-scales)")
    config_choice = Prompt.ask("[cyan]Config[/cyan]", choices=["1", "2", "3", "4"], default="2")
    
    config_map = {"1": (16, 64, 2), "2": (32, 128, 3), "3": (64, 256, 4)}
    if config_choice == "4":
        custom_agents = Prompt.ask("[cyan]Agent count (or 'auto')[/cyan]", default="128")
        if custom_agents.lower() == "auto":
            custom_agents = "128"
        agents = int(custom_agents)
        auto = auto_scale(agents)
        agent_dim = auto["agent_dim"]
        steps = auto["steps"]
        console.print(f"[dim]Auto-scaled: agents={agents} -> dim={agent_dim}, steps={steps}[/dim]")
    else:
        agents, agent_dim, steps = config_map[config_choice]
    
    console.print("\n[bold cyan]3.[/bold cyan] Insert after layers")
    console.print("    [1] Layer 5")
    console.print("    [2] Layer 10")
    console.print("    [3] All layers")
    console.print("    [4] Custom")
    layer_choice = Prompt.ask("[cyan]Layers[/cyan]", choices=["1", "2", "3", "4"], default="1")
    layer_map = {"1": [5], "2": [10], "3": "all"}
    if layer_choice == "4":
        layers = [int(x) for x in Prompt.ask("[cyan]Layers[/cyan]", default="5").split(",")]
    elif layer_choice == "3":
        layers = "all"
    else:
        layers = layer_map[layer_choice]
    
    console.print()
    
    # Summary
    summary = Table(box=None, pad_edge=True)
    summary.add_column("", style="cyan")
    summary.add_column("", style="white")
    summary.add_row("HF Model", hf_model)
    summary.add_row("Agents", str(agents))
    summary.add_row("Agent Dim", str(agent_dim))
    summary.add_row("Steps", str(steps))
    summary.add_row("Layers", str(layers))
    
    console.print(Panel(summary, title="[bold green]Integration Config[/bold green]", border_style="green"))
    console.print()
    
    if Confirm.ask("[green]Integrate now?[/green]", default=False):
        run_integration(hf_model, agents, agent_dim, steps, layers)
    
    Prompt.ask("[dim]Press Enter...[/dim]", default="")


def run_integration(model_name, agents, agent_dim, steps, layers):
    """Actually run the integration"""
    console.print("[dim]Loading model...[/dim]")
    
    try:
        import torch
        from globular import GlobularIntegrationConfig, apply_globular_to_model, count_parameters
        from globular.hf_utils import safe_load_causal_lm, safe_load_tokenizer, ensure_tokenizer_padding
        
        # Load model
        model = safe_load_causal_lm(model_name, device_map="cpu", torch_dtype=torch.float32)
        tokenizer = ensure_tokenizer_padding(safe_load_tokenizer(model_name))
        
        base_params = count_parameters(model)
        console.print(f"[green]Base model: {base_params:,} params[/green]")
        
        # Create config
        insert_layers = layers if layers != "all" else list(range(model.config.num_hidden_layers))
        
        config = GlobularIntegrationConfig(
            insert_after_layers=insert_layers,
            num_agents=agents,
            agent_dim=agent_dim,
            steps=steps,
            model_dim=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
        )
        
        # Apply
        console.print(f"[dim]Applying Globular after {insert_layers}...[/dim]")
        model = apply_globular_to_model(model, config)
        
        total_params = count_parameters(model)
        added = total_params - base_params
        
        console.print(Panel.fit(
            f"[bold green]Integration Complete![/bold green]\n\n"
            f"Base:     {base_params:,}\n"
            f"Added:    {added:,}\n"
            f"Total:    {total_params:,}",
            border_style="green"
        ))
        
        # Save option
        if Confirm.ask("[green]Save model?[/green]", default=False):
            output = Prompt.ask("[cyan]Output directory[/cyan]", default="./output/globular-model")
            console.print(f"[dim]Saving to {output}...[/dim]")
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            console.print("[green]Saved![/green]")
        
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Install dependencies: pip install transformers torch[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


# =============================================================================
# TRAIN
# =============================================================================

def train_flow():
    console.clear()
    header("TRAIN")
    
    console.print("\n[bold cyan]1.[/bold cyan] Training type")
    console.print("    [1] From Scratch (new Globular)")
    console.print("    [2] Fine-tune existing model")
    console.print("    [3] Instruct Training (Opencode-Instruct)")
    train_type = Prompt.ask("[cyan]Type[/cyan]", choices=["1", "2", "3"], default="1")
    
    console.print("\n[bold cyan]2.[/bold cyan] GPU")
    console.print("    [1] Local (RTX 3050 - 8GB)")
    console.print("    [2] Colab (T4 - 16GB)")
    console.print("    [3] Colab (A100 - 40GB)")
    gpu_choice = Prompt.ask("[cyan]GPU[/cyan]", choices=["1", "2", "3"], default="1")
    
    gpu_map = {"1": ("RTX 3050", 8), "2": ("T4", 16), "3": ("A100", 40)}
    gpu_name, gpu_mem = gpu_map[gpu_choice]
    
    console.print("\n[bold cyan]3.[/bold cyan] Preset")
    for k, p in PRESETS.items():
        console.print(f"    [{k}] {p['name']} ({p['focus']})")
    preset_choice = Prompt.ask("[cyan]Preset[/cyan]", choices=list(PRESETS.keys()), default="1")
    preset = PRESETS[preset_choice]
    
    console.print("\n[bold cyan]4.[/bold cyan] Dataset")
    console.print("    [1] Use HF dataset")
    console.print("    [2] Use local dataset")
    console.print("    [3] Generate synthetic")
    dataset_choice = Prompt.ask("[cyan]Dataset[/cyan]", choices=["1", "2", "3"], default="1")
    
    if dataset_choice == "1":
        dataset = Prompt.ask("[cyan]Dataset name[/cyan]", default="open-web-math")
    elif dataset_choice == "2":
        dataset = Prompt.ask("[cyan]Dataset path[/cyan]", default="./data/train.jsonl")
    else:
        dataset = "synthetic"
    
    console.print("\n[bold cyan]5.[/bold cyan] Training config")
    epochs = Prompt.ask("[cyan]Epochs[/cyan]", default="3")
    batch_size = Prompt.ask("[cyan]Batch size[/cyan]", default="4")
    seq_len = Prompt.ask("[cyan]Sequence length[/cyan]", default="128" if gpu_choice == "1" else "256")
    max_samples = Prompt.ask("[cyan]Max samples[/cyan]", default="1000")
    lr = Prompt.ask("[cyan]Learning rate[/cyan]", default="1e-4")
    
    # Summary
    console.print()
    summary = Table(box=None, pad_edge=True)
    summary.add_column("", style="cyan")
    summary.add_column("", style="white")
    summary.add_row("Type", ["From Scratch", "Fine-tune", "Instruct"][int(train_type) - 1])
    summary.add_row("GPU", f"{gpu_name} ({gpu_mem}GB)")
    summary.add_row("Preset", preset["name"])
    summary.add_row("Dataset", dataset)
    summary.add_row("Epochs", epochs)
    summary.add_row("Batch", batch_size)
    summary.add_row("Seq Len", seq_len)
    summary.add_row("Samples", max_samples)
    summary.add_row("LR", lr)
    
    console.print(Panel(summary, title="[bold green]Training Config[/bold green]", border_style="green"))
    console.print()
    
    if Confirm.ask("[green]Start training?[/green]", default=False):
        run_training(train_type, gpu_name, preset, dataset, epochs, batch_size, lr, seq_len, max_samples)


def run_training(train_type, gpu, preset, dataset, epochs, batch_size, lr, seq_len=128, max_samples=1000):
    """Actually run training"""
    console.print(f"[dim]Running training...[/dim]")
    
    # Import training module
    try:
        from globular.train import TrainingConfig, train_globular
        
        # Map preset to training args
        preset_map = {"1": "lambda", "2": "opencode", "3": "opencode_reason"}
        
        config = TrainingConfig(
            agents=preset["agents"],
            agent_dim=preset["dim"],
            steps=preset["steps"],
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            max_seq_len=int(seq_len),
            max_samples=int(max_samples),
            dataset=dataset,
            output_dir="./output/globular",
        )
        
        train_globular(config)
        
    except ImportError as e:
        console.print(f"[red]Missing: {e}[/red]")
        console.print("[yellow]Install: pip install torch datasets accelerate[/yellow]")
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


# =============================================================================
# PUBLISH
# =============================================================================

def publish_flow():
    console.clear()
    header("PUBLISH")
    
    console.print("\n[bold cyan]1.[/bold cyan] Destination")
    console.print("    [1] HuggingFace Hub")
    console.print("    [2] GGUF (llama.cpp)")
    console.print("    [3] Local directory")
    dest_choice = Prompt.ask("[cyan]Destination[/cyan]", choices=["1", "2", "3"], default="1")
    
    if dest_choice == "1":
        # HF Hub
        run_publish_hf()
    elif dest_choice == "2":
        # GGUF
        run_export_gguf()
    else:
        # Local
        run_save_local()
    
    Prompt.ask("[dim]Press Enter...[/dim]", default="")


def run_publish_hf():
    console.print("\n[bold cyan]1.[/bold cyan] Model to publish")
    
    try:
        from huggingface_hub import whoami
        user = whoami()
        console.print(f"[green]Logged in as: {user['name']}[/green]")
    except:
        console.print("[yellow]Not logged in to HF[/yellow]")
        if Confirm.ask("[green]Login now?[/green]", default=True):
            token = Prompt.ask("[cyan]HF token (hidden)[/cyan]", password=True)
            try:
                from huggingface_hub import login
                login(token=token)
                console.print("[green]Logged in![/green]")
            except:
                console.print("[red]Login failed[/red]")
                return
    
    repo_id = Prompt.ask("[cyan]Repository ID[/cyan]", default="username/globular-model")
    
    console.print("\n[bold cyan]2.[/bold cyan] Settings")
    console.print("    [1] Public")
    console.print("    [2] Private")
    vis_choice = Prompt.ask("[cyan]Visibility[/cyan]", choices=["1", "2"], default="1")
    visibility = "public" if vis_choice == "1" else "private"
    
    console.print()
    if Confirm.ask("[green]Publish to HF Hub?[/green]", default=False):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", private=vis_choice == "2", exist_ok=True)
            
            console.print(f"[yellow]Upload not yet fully implemented[/yellow]")
            console.print(f"[dim]Run: huggingface-cli upload[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def run_export_gguf():
    console.print("\n[bold cyan]1.[/bold cyan] Model to convert")
    model_path = Prompt.ask("[cyan]Model path or HF ID[/cyan]", default="./output/globular-model")
    
    console.print("\n[bold cyan]2.[/bold cyan] Quantization")
    console.print("    [1] Q4_K (recommended)")
    console.print("    [2] Q5_K")
    console.print("    [3] Q8_0")
    console.print("    [4] F16")
    quant_choice = Prompt.ask("[cyan]Quantization[/cyan]", choices=["1", "2", "3", "4"], default="1")
    quant_map = {"1": "q4_k", "2": "q5_k", "3": "q8_0", "4": "f16"}
    quant = quant_map[quant_choice]
    
    output_path = Prompt.ask("[cyan]Output path[/cyan]", default="./output/globular-model.gguf")
    
    console.print()
    if Confirm.ask("[green]Export to GGUF?[/green]", default=False):
        console.print(f"[yellow]GGUF export not implemented[/yellow]")
        console.print("[dim]Use llama.cpp directly:[/dim]")
        console.print(f"  llama-quantize {model_path} {output_path} {quant}")


def run_save_local():
    console.print("\n[bold cyan]1.[/bold cyan] Model to save")
    model_path = Prompt.ask("[cyan]Model path or HF ID[/cyan]", default="Qwen/Qwen2-0.5B")
    output_dir = Prompt.ask("[cyan]Output directory[/cyan]", default="./output/globular-model")
    
    console.print()
    if Confirm.ask("[green]Save locally?[/green]", default=False):
        try:
            import torch
            from globular.hf_utils import safe_load_causal_lm, safe_load_tokenizer, ensure_tokenizer_padding
            
            console.print(f"[dim]Loading {model_path}...[/dim]")
            model = safe_load_causal_lm(model_path, device_map="cpu")
            tokenizer = ensure_tokenizer_padding(safe_load_tokenizer(model_path))
            
            console.print(f"[dim]Saving to {output_dir}...[/dim]")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            console.print("[green]Saved![/green]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# =============================================================================
# HF SEARCH
# =============================================================================

def hf_search_flow():
    console.clear()
    header("HF SEARCH")
    
    console.print("\n[bold cyan]1.[/bold cyan] Search type")
    console.print("    [1] Models")
    console.print("    [2] Datasets")
    search_type = Prompt.ask("[cyan]Type[/cyan]", choices=["1", "2"], default="1")
    
    search = Prompt.ask("[cyan]Search term[/cyan]", default="qwen" if search_type == "1" else "code reasoning")
    limit = Prompt.ask("[cyan]Max results[/cyan]", default="10")
    
    console.print()
    console.print(f"[dim]Searching...[/dim]")
    
    if not HF_AVAILABLE:
        console.print("[red]huggingface_hub not installed")
        Prompt.ask("[dim]Press Enter...[/dim]", default="")
        return
    
    try:
        if search_type == "1":
            results = list(list_models(search=search, limit=int(limit)))
        else:
            results = list(list_datasets(search=search, limit=int(limit)))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        Prompt.ask("[dim]Press Enter...[/dim]", default="")
        return
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        Prompt.ask("[dim]Press Enter...[/dim]", default="")
        return
    
    # Table
    table = Table(show_header=True, header_style="bold cyan" if search_type == "1" else "bold green", 
                border_style="cyan" if search_type == "1" else "green", pad_edge=True)
    table.add_column("Name", style="cyan" if search_type == "1" else "green")
    table.add_column("Downloads", justify="right", style="yellow")
    
    for r in results:
        name = getattr(r, 'id', 'unknown') or 'unknown'
        if len(name) > 50:
            name = name[:47] + "..."
        downloads = getattr(r, 'downloads', 0) or 0
        table.add_row(name, f"{downloads:,}")
    
    label = "Models" if search_type == "1" else "Datasets"
    console.print(Panel(table, title=f"[bold]{len(results)} {label}[/bold]", 
                       border_style="cyan" if search_type == "1" else "green"))
    console.print()
    Prompt.ask("[dim]Press Enter...[/dim]", default="")


# =============================================================================
# SETTINGS
# =============================================================================

def settings_flow():
    console.clear()
    header("SETTINGS")
    
    console.print("\n[bold cyan]1.[/bold cyan] HF Login")
    try:
        from huggingface_hub import whoami
        user = whoami()
        console.print(f"[green]Logged in as: {user['name']}[/green]")
    except:
        console.print("[yellow]Not logged in[/yellow]")
    
    if Confirm.ask("[green]Login to HF?[/green]", default=False):
        token = Prompt.ask("[cyan]HF token[/cyan]", password=True)
        login(token=token)
        console.print("[green]Logged in![/green]")
    
    console.print("\n[bold cyan]2.[/bold cyan] Default settings")
    console.print("    Model:      Qwen2-0.5B")
    console.print("    Preset:     Lambda")
    console.print("    Size:       Standard")
    
    console.print("\n[bold cyan]3.[/bold cyan] GPU Info")
    console.print("    Local:     RTX 3050 (8GB)")
    console.print("    Colab T4:  16GB")
    console.print("    Colab A100: 40GB")
    
    console.print()
    Prompt.ask("[dim]Press Enter...[/dim]", default="")


# =============================================================================
# MAIN
# =============================================================================

def main():
    while True:
        choice = main_menu()
        
        if choice == "q":
            console.print("\n[cyan]Thanks![/cyan]\n")
            break
        elif choice == "1":
            simple_flow()
        elif choice == "2":
            expert_flow()
        elif choice == "3":
            integrate_flow()
        elif choice == "4":
            train_flow()
        elif choice == "5":
            publish_flow()
        elif choice == "6":
            hf_search_flow()
        elif choice == "6":
            settings_flow()


if __name__ == "__main__":
    main()
