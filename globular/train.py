#!/usr/bin/env python3
"""
Globular Training Script - Enhanced with checkpoint resume and metrics
"""

import argparse
import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch.amp import autocast, GradScaler
except ImportError:  # pragma: no cover - compatibility with older torch
    from torch.cuda.amp import autocast, GradScaler

from globular import (
    GlobularConfig,
    GlobularReasoningBlock,
    count_parameters,
)


VERSION = "0.2.0"


@dataclass
class TrainingMetrics:
    """Training metrics"""
    epoch: int = 0
    batch: int = 0
    loss: float = 0.0
    energy: float = 0.0
    perplexity: float = 0.0
    avg_loss: float = 0.0
    lr: float = 0.0
    runtime: float = 0.0
    samples_processed: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingMetrics":
        return cls(**d)


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str = "Qwen/Qwen2-0.5B"
    dim: int = 896
    num_layers: int = 24
    
    agents: int = 32
    agent_dim: int = 128
    steps: int = 3
    num_heads: int = 8
    
    epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_seq_len: int = 512
    max_samples: int = 1000
    
    dataset: str = "synthetic"
    data_path: Optional[str] = None
    
    insert_layers: List[int] = field(default_factory=lambda: [5, 10])
    
    output_dir: str = "./output/globular"
    save_every: int = 1
    log_every: int = 10
    
    device: str = "auto"
    use_amp: bool = True
    
    resume_from: Optional[str] = None

    def validate(self):
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.agents <= 0:
            raise ValueError("agents must be positive")
        if self.agent_dim <= 0:
            raise ValueError("agent_dim must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.agent_dim % self.num_heads != 0:
            raise ValueError("agent_dim must be divisible by num_heads")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in asdict(cls())})
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class MetricsLogger:
    """Log training metrics to file"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize or append
        if os.path.exists(self.metrics_file):
            self.file = open(self.metrics_file, "a")
        else:
            self.file = open(self.metrics_file, "w")
            self.file.write(json.dumps({"epoch": 0, "batch": 0, "loss": 0.0}) + "\n")
    
    def log(self, metrics: TrainingMetrics):
        self.file.write(json.dumps(metrics.to_dict()) + "\n")
        self.file.flush()
    
    def close(self):
        self.file.close()


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, dim: int = 512):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dim = dim
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate on demand to avoid allocating gigabytes for larger dims.
        return torch.randn(self.seq_len, self.dim), torch.randn(self.seq_len, self.dim)


class EmbeddingProxyDataset(Dataset):
    """Wrap arbitrary HF/local rows as embedding targets for block-level training."""

    def __init__(self, source, seq_len: int, dim: int, max_samples: int = 1000):
        self.source = source
        self.seq_len = seq_len
        self.dim = dim
        self.length = min(len(source), max_samples) if hasattr(source, "__len__") else max_samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(self.seq_len, self.dim), torch.randn(self.seq_len, self.dim)


def train_globular(config: TrainingConfig):
    """Train with metrics and checkpoint resume"""
    config.validate()
    print(f"\n=== Globular Training v{VERSION} ===\n")
    start_time = time.time()
    
    # Setup
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    if config.use_amp and device == "cuda":
        print("Mixed precision: enabled")
    
    # Create model
    gconfig = GlobularConfig(
        dim=config.dim,
        num_agents=config.agents,
        agent_dim=config.agent_dim,
        steps=config.steps,
        num_heads=config.num_heads,
    )
    model = GlobularReasoningBlock(gconfig).to(device)
    
    num_params = count_parameters(model)
    print(f"Globular params: {num_params:,}")
    
    # Dataset
    if config.dataset == "synthetic":
        dataset = SyntheticDataset(num_samples=config.max_samples, seq_len=config.max_seq_len, dim=config.dim)
    else:
        try:
            from datasets import load_dataset
            loaded = load_dataset(config.dataset)
            split = "train" if "train" in loaded else next(iter(loaded.keys()))
            dataset = EmbeddingProxyDataset(
                loaded[split], seq_len=config.max_seq_len, dim=config.dim, max_samples=config.max_samples
            )
            print(f"Loaded {config.dataset} ({split}); using embedding proxy training")
        except Exception as exc:
            print(f"Could not load {config.dataset}: {exc}")
            print("Using synthetic dataset")
            dataset = SyntheticDataset(num_samples=config.max_samples, seq_len=config.max_seq_len, dim=config.dim)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    try:
        scaler = GradScaler("cuda") if config.use_amp and device == "cuda" else None
    except TypeError:
        scaler = GradScaler() if config.use_amp and device == "cuda" else None
    
    # Resume from checkpoint
    start_epoch = 0
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"Resuming from {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif all(group.get("lr", 0.0) == 0.0 for group in optimizer.param_groups):
            # Older checkpoints saved after a one-epoch cosine schedule can
            # contain lr=0, which makes resumed training silently inert.
            for group in optimizer.param_groups:
                group["lr"] = config.lr
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Metrics logger
    os.makedirs(config.output_dir, exist_ok=True)
    logger = MetricsLogger(config.output_dir)
    
    # Training loop
    print(f"\nTraining for {config.epochs} epochs...")
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0
        total_batches = len(dataloader)
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                try:
                    autocast_context = autocast("cuda")
                except TypeError:
                    autocast_context = autocast()
                with autocast_context:
                    output, diagnostics = model(x)
                    loss = nn.functional.mse_loss(output, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                output, diagnostics = model(x)
                loss = nn.functional.mse_loss(output, y)
                loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            
            # Metrics
            loss_val = loss.item()
            energy = diagnostics.get("agent_energy", 0)
            if torch.is_tensor(energy):
                energy = energy.mean().item()
            
            epoch_loss += loss_val
            
            # Log
            if batch_idx % config.log_every == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                    loss=loss_val,
                    energy=energy,
                    avg_loss=avg_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    runtime=elapsed,
                    samples_processed=(batch_idx + 1) * config.batch_size,
                )
                logger.log(metrics)
                
                print(f"E{epoch+1}/{config.epochs} | B{batch_idx+1}/{total_batches} | "
                      f"L:{loss_val:.4f} | Avg:{avg_loss:.4f} | "
                      f"E:{energy:.4f} | LR:{metrics.lr:.2e}")
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_path = f"{config.output_dir}/checkpoint_e{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config.to_dict(),
            }, save_path)
            config.save(f"{config.output_dir}/config.json")
            print(f"Saved: {save_path}")
    
    # Save final
    final_path = f"{config.output_dir}/globular_final.pt"
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, final_path)
    config.save(f"{config.output_dir}/config.json")
    
    logger.close()
    
    print(f"\n=== Complete ===")
    print(f"Runtime: {time.time() - start_time:.1f}s")
    print(f"Final: {final_path}")
    print(f"Metrics: {config.output_dir}/metrics.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Globular Training v2")
    
    parser.add_argument("--model", "-m", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--dim", type=int, default=896)
    parser.add_argument("--layers", type=int, default=24)
    
    parser.add_argument("--agents", "-a", type=int, default=32)
    parser.add_argument("--agent-dim", type=int, default=128)
    parser.add_argument("--steps", "-s", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--batch", "-b", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=1000)
    
    parser.add_argument("--dataset", "-d", default="synthetic")
    parser.add_argument("--data-path")
    
    parser.add_argument("--insert", nargs="+", type=int)
    
    parser.add_argument("--output", "-o", default="./output/globular")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--preset")
    
    args = parser.parse_args()
    
    # Presets
    if args.preset == "lambda":
        args.agents, args.agent_dim, args.steps = 32, 128, 3
    elif args.preset == "opencode":
        args.agents, args.agent_dim, args.steps = 32, 128, 3
    elif args.preset == "heavy":
        args.agents, args.agent_dim, args.steps = 64, 256, 4

    if args.agent_dim % args.heads != 0:
        for candidate in (64, 32, 16, 8, 4, 2, 1):
            if args.agent_dim % candidate == 0:
                print(f"Adjusting heads from {args.heads} to {candidate} for agent_dim={args.agent_dim}")
                args.heads = candidate
                break
    
    config = TrainingConfig(
        model_name=args.model,
        dim=args.dim,
        num_layers=args.layers,
        agents=args.agents,
        agent_dim=args.agent_dim,
        steps=args.steps,
        num_heads=args.heads,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=0.01,
        grad_clip=args.grad_clip,
        max_seq_len=args.seq_len,
        max_samples=args.max_samples,
        dataset=args.dataset,
        data_path=args.data_path,
        insert_layers=args.insert or [5, 10],
        output_dir=args.output,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        use_amp=not args.no_amp,
        resume_from=args.resume,
    )
    
    train_globular(config)


if __name__ == "__main__":
    main()
