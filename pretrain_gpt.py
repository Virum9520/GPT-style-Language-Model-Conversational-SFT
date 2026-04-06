"""
EECS 595 HW3: GPT Pretraining Script

This script contains the complete training loop for pretraining a GPT model.
Students need to implement the core components in gpt.py before running this script.

Usage:
    python pretrain_gpt.py

The script will:
1. Load data from the specified dataset
2. Create train/validation splits
3. Initialize the GPT model
4. Train the model with mixed precision
5. Save checkpoints and log to wandb

TODO: Students need to implement the following components in gpt.py:
- GPTEmbedding: Token embeddings (no positional embeddings needed)
- MultiHeadAttention: Attention mechanism with RoPE
- SwiGLU: Modern activation function
- FeedForward: Position-wise MLP
- TransformerBlock: Combines attention and MLP
- GPTModel: Complete GPT model
- Dataset classes: Data loading utilities
"""

import os
import math
import numpy as np
import random
import logging
import argparse
from typing import Optional, Callable, List, Tuple, Dict, Any

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler

# Data loading imports
from torch.utils.data import Dataset, DataLoader
import json
import glob
import gzip
import bz2
import datetime

# Arrow dataset support
from datasets import load_from_disk

# Tokenization imports
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Progress and timing
from tqdm.auto import tqdm, trange
import time
import wandb

# Import our GPT implementation
import gpt

# Set CuPy/CUDA to allow TF32 computations
# This can provide a speedup on compatible GPUs (RTX 4000 series, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT model')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                       default='/shared/0/projects/teaching/eecs595/data/fineweb-edu-sample-1B.jsonl.gz',
                       help='Path to the training data (JSONL.gz file or Arrow dataset directory)')
    parser.add_argument('--data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of training data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--max_docs', type=int, default=None,
                       help='Maximum number of documents to load (for testing, only applies to raw text)')

    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='Vocabulary size (auto-detected if not specified)')
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--emb_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=2,
                       help='Maximum number of epochs')
    parser.add_argument('--target_tokens', type=int, default=1_200_000_000,
                       help='Target number of tokens to train on')

    # Validation arguments
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to validation data')
    parser.add_argument('--eval_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of validation data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--eval_max_docs', type=int, default=None,
                       help='Maximum number of documents to load for validation (only for raw text)')
    parser.add_argument('--eval_max_docs_step', type=int, default=None,
                       help='Maximum number of validation documents to use during step evaluation (None = use all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Validation batch size')
    parser.add_argument('--eval_max_docs_epoch', type=int, default=None,
                       help='Maximum number of validation batches to use at end-of-epoch evaluation (None = use all)')


    # Logging and saving
    parser.add_argument('--output_dir', type=str,
                       default='/shared/0/projects/teaching/eecs595/models/pico-gpt/pretrained-models/',
                       help='Output directory for saving models')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                       help='Evaluate model every N steps')
    parser.add_argument('--wandb_project', type=str, default='gpt-pretraining',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str,
                       default=f"gpt-pretraining-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                       help='Wandb run name')
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """Determine the best available device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    else:
        return device_arg

def get_amp_dtype(device):
    '''Get the appropriate AMP dtype for mixed precision training on the device.'''

    if device.startswith('cuda'):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == 'mps':
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32  # or disable autocast on CPU
    return amp_dtype

def load_data(data_path, max_docs=None, data_format='jsonl'):
    """
    Load data from JSONL file or Arrow dataset.

    Args:
        data_path: Path to the data file or Arrow dataset directory
        max_docs: Maximum number of documents to load (only for raw text)
        data_format: Format of the data ('jsonl' or 'arrow')
    Returns:
        List of text documents (for raw text) or None (for Arrow datasets)
    """
    if data_format == 'arrow':
        print(f"Using Arrow dataset from {data_path}")
        # For Arrow datasets, we don't need to load the data here
        # The GPTArrowDataset in gpt.py will handle loading
        return None
    else:
        print(f"Loading data from {data_path}")

        ofunc = gzip.open if data_path.endswith('gz') else open
        docs = []

        with ofunc(data_path, 'rt') as f:
            for i, line in enumerate(tqdm(f, desc="Reading data from file")):
                if max_docs and i >= max_docs:
                    break
                docs.append(json.loads(line)['text'])

        print(f"Loaded {len(docs)} documents")
        return docs


def create_dataloaders(docs, tokenizer, config, args):
    """Create train and validation dataloaders."""
    print("Creating dataloaders...")

    ###########################################################################
    #                            TODO 2.1: YOUR CODE HERE                         #
    #                                                                         #
    # Implement dataloader creation for training:                           #
    #                                                                         #
    # 1. Check if using Arrow dataset format (args.data_format == 'arrow')   #
    # 2. If Arrow format:                                                     #
    #    - Use gpt.create_dataloader() with arrow_dataset_path=args.data_path #
    #    - Create both train and val loaders using the same Arrow dataset     #
    #    - Note: Arrow datasets are typically pre-split or can be split       #
    # 3. If raw text format:                                                  #
    #    - Split documents into trainvalidation sets (95%/5%)               #
    #    - Create training DataLoader using docs                              #
    #    - Create validation DataLoader using validation docs                 #
    # 4. Print dataset statistics                                            #
    # 5. Return both dataloaders                                              #
    #                                                                         #
    # Proper data splitting is essential for evaluation!                     #
    ###########################################################################

    # 1) Arrow dataset mode
    if args.data_format == 'arrow':
        # Train loader from Arrow
        train_loader = gpt.create_dataloader(
            arrow_dataset_path=args.data_path,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
        # Validation loader: support Arrow or JSONL based on eval_data_format
        if args.eval_data_path:
            if getattr(args, "eval_data_format", "jsonl") == "arrow":
                val_loader = gpt.create_dataloader(
                    arrow_dataset_path=args.eval_data_path,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.num_workers,
                )
            else:
                # Load raw text validation docs (JSONL)
                eval_docs = load_data(args.eval_data_path, args.eval_max_docs, data_format="jsonl")
                val_loader = gpt.create_dataloader(
                    txt=eval_docs,
                    batch_size=args.eval_batch_size,
                    max_length=config['context_length'],
                    stride=config['context_length'] // 2,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.num_workers,
                )
        else:
            # Fallback: validate on the same Arrow dataset
            val_loader = gpt.create_dataloader(
                arrow_dataset_path=args.data_path,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )
    else:
        # 2) Raw text format: split into 95% train / 5% val
        assert docs is not None, "Raw text format requires loaded documents"
        n_docs = len(docs)
        split = int(0.95 * n_docs)
        train_docs = docs[:split]
        val_docs = docs[split:] if split < n_docs else docs[:1]  # ensure non-empty

        train_loader = gpt.create_dataloader(
            txt=train_docs,
            batch_size=args.batch_size,
            max_length=config['context_length'],
            stride=config['context_length'] // 2,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
        val_loader = gpt.create_dataloader(
            txt=val_docs,
            batch_size=args.eval_batch_size,
            max_length=config['context_length'],
            stride=config['context_length'] // 2,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )

    # 4) Print dataset statistics
    train_len = len(train_loader.dataset) if hasattr(train_loader, "dataset") else -1
    val_len = len(val_loader.dataset) if hasattr(val_loader, "dataset") else -1
    print(f"Train samples: {train_len} | Val samples: {val_len}")

    print("✅ Dataloaders created")
    return train_loader, val_loader


def evaluate_validation_loss(model, val_loader, loss_fn, device, max_docs=None):
    """Evaluate the model's loss on the validation dataset.

    Args:
        model: The GPT model to evaluate
        val_loader: Validation data loader
        loss_fn: Loss function to use
        device: Device to run evaluation on
        max_docs: Maximum number of validation batches to process (None = use all)
    """
    ###########################################################################
    #                            TODO 2.2: YOUR CODE HERE                     #
    #                                                                         #
    # Implement validation loss evaluation:                                   #
    #                                                                         #
    # 1. Set model to evaluation mode (model.eval())                          #
    # 2. Initialize loss tracking variables                                   #
    # 3. Iterate through validation batches with torch.no_grad():             #
    #    - Move data to device                                                #
    #    - Forward pass with mixed precision (optional but recommended)       #
    #    - Compute loss and accumulate                                        #
    #    - Stop early if max_docs limit is reached                            #
    # 4. Calculate average validation loss                                    #
    # 5. Set model back to training mode (model.train())                      #
    # 6. Return the average validation loss                                   #
    #                                                                         #
    # Note: max_docs parameter allows limiting validation batches for faster  #
    # step evaluation, while end-of-epoch evaluation uses all validation data #
    # This is crucial for monitoring overfitting during training!             #
    ###########################################################################

    model.eval()
    total_loss = 0.0
    count = 0

    amp_dtype = get_amp_dtype(device)
    device_type = 'cuda' if device.startswith('cuda') else ('mps' if device == 'mps' else 'cpu')

    with torch.no_grad():
        for b_idx, (input_ids, labels) in enumerate(val_loader):
            if max_docs is not None and b_idx >= max_docs:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type != 'cpu')):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(1, count)
    model.train()
    return avg_loss

def train_model(model, train_loader, val_loader, config, args):
    """Train the GPT model."""
    device = get_device(args.device)
    amp_dtype = get_amp_dtype(device)
    print(f"Using device: {device}")

    # Move model to device
    model.to(device)

    # Initialize training components
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )


    # Calculate training steps
    tokens_per_step = config['context_length'] * args.batch_size
    total_steps = math.ceil(args.target_tokens / tokens_per_step)
    warmup_steps = min(400, int(0.02 * total_steps))  # ~2% warmup, capped at 400

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,  # half-cosine
    )

    # Initialize wandb
    #
    # NOTE: If you're doing any other customization to your model design, we
    # recommend logging these configuration details to wandb for easier analysis
    # on whether the changes you made are helping or hurting performance.
    wandb_config = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "position_embedding": "rope",
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "context_length": config["context_length"],
        "drop_rate": config["drop_rate"],
    }
    wandb.init(project=args.wandb_project,
               config=wandb_config,
               name=args.wandb_run_name,
               )

    # Training loop
    model.train()
    opt_step = 0
    global_step = 0  # Track total steps across all epochs
    losses = []

    # Track last executed steps to prevent duplicate evaluation/saving
    last_eval_step = -1
    last_save_step = -1


    # Normally, we want to use a large batch size to get better gradient estimates.
    # However, if we use a large batch size, we will run out of memory. Therefore,
    # we'll use a technique called gradient accumulation to simulate a larger batch size.
    # We'll still use batches of a certain size, but we won't call the optimizer.step()
    # after each batch. Instead, we'll accumulate gradients over multiple batches
    # and call the optimizer.step() after a certain number of batches. You'll see the smaller batch size
    # called "micro-batch" in the code (and in practice) and the larger batch size called
    # the effective batch size or macro-batch.
    #
    # GRADIENT ACCUMULATION EXPLANATION:
    # Gradient accumulation allows us to simulate larger batch sizes by:
    # - Computing gradients on smaller "micro-batches"
    # - Accumulating gradients across multiple micro-batches
    # - Only updating parameters after accumulating gradients from 'accum' batches
    # - This enables training with effective batch size = micro_batch_size * accum
    # - Example: micro_batch=32, accum=8 → effective batch_size=256
    # - Benefits: Better gradient estimates, memory efficiency, stable training

    # Gradient accumulation variables
    target_global_batch = 256
    micro_batch = args.batch_size
    accum = max(1, target_global_batch // micro_batch)

    print(f"Starting training...")
    print(f"Gradient accumulation steps: {accum}")


    ###########################################################################
    #                            TODO 2.3: YOUR CODE HERE                         #
    #
    # Students need to implement the core training loop
    # This involves:
    # 1. Computing gradients with loss.backward() (scaled by accumulation factor)
    # 2. Gradient clipping to prevent exploding gradients
    # 3. Optimizer step to update parameters (only every 'accum' steps)
    # 4. Learning rate scheduling
    # 5. Zeroing gradients for next iteration
    # 6. Gradient accumulation for larger effective batch sizes
    # 7. Saving the model model according to the save_every step
    # 8. Evaluating the model on the validation set according to
    #    the eval_every step and using the eval_max_docs_step docs (if a validation dataset is provided)
    # 9. Logging the loss to wandb
    # 10. Saving the model at the end of each epoch
    # 11. Logging the full validation loss to wandb at the end of each epoch
    #
    #
    # CORE STEPS:
    # 1. Scale loss by accumulation factor: (loss / accum).backward()
    # 2. Check if we've accumulated enough gradients: if (step + 1) % accum == 0
    # 3. Clip gradients to prevent explosion: torch.nn.utils.clip_grad_norm_()
    # 4. Update parameters: optimizer.step()
    # 5. Update learning rate: scheduler.step()
    # 6. Clear gradients: optimizer.zero_grad()
    # 7. Track optimization steps: opt_step += 1, global_step += 1
    #
    ###########################################################################

    scaler = GradScaler(enabled=(device.startswith('cuda') and amp_dtype == torch.float16))
    device_type = 'cuda' if device.startswith('cuda') else ('mps' if device == 'mps' else 'cpu')

    for epoch in trange(args.max_epochs, desc="Epoch"):

        for step, (input_ids, labels) in enumerate(tqdm(train_loader, position=1, leave=True, desc="Step")):


            # NOTE: Students need to implement the forward pass
            # This involves:
            # 1. Moving input_ids and labels to the correct device
            # 2. Calling model(input_ids) to get logits
            # 3. Computing loss using CrossEntropyLoss
            # 4. Handling mixed precision training with autocast

            # NOTE: See https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
            # for more information on mixed precision training if you're curious!
            # We strongly recommend using mixed precision training for faster training and reduced memory usage.

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type != 'cpu')):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Gradient accumulation
            loss_to_backprop = loss / accum
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            # Optimization step every 'accum' micro-batches
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                opt_step += 1
                global_step += 1

                # Log training loss
                wandb.log({"train/loss": loss.item(), "opt_step": opt_step, "global_step": global_step})

                # Periodic evaluation
                if args.eval_every and (global_step % args.eval_every == 0) and (global_step != last_eval_step):
                    if val_loader is not None:
                        val_loss = evaluate_validation_loss(
                            model, val_loader, loss_fn, device, max_docs=args.eval_max_docs_step
                        )
                        wandb.log({"val/loss_step": val_loss, "global_step": global_step})
                    last_eval_step = global_step

                # Periodic saving
                if args.save_every and (global_step % args.save_every == 0) and (global_step != last_save_step):
                    save_path = os.path.join(args.output_dir, f"model_step_{global_step}.pt")
                    torch.save({"model_state_dict": model.state_dict(),
                                "config": config,
                                "opt_step": opt_step,
                                "global_step": global_step}, save_path)
                    last_save_step = global_step

        # Save model at end of epoch
        epoch_save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "global_step": global_step}, epoch_save_path)

        # Final evaluation for epoch (optionally limit number of batches)
        if val_loader is not None:
            full_val_loss = evaluate_validation_loss(
                model, val_loader, loss_fn, device, max_docs=args.eval_max_docs_epoch
            )
            wandb.log({"val/loss_epoch": full_val_loss, "epoch": epoch + 1, "global_step": global_step})


    print("Training completed!")
    wandb.finish()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print("Setting up tokenizer...")
    tokenizer = gpt.setup_tokenizer()

    # Determine vocabulary size
    if args.vocab_size is None:
        special_tokens = ["<|user|>", "<|assistant|>", "<|end|>", "<|system|>", "<|pad|>"]
        max_token_id = max(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
        vocab_size = max_token_id + 1
    else:
        vocab_size = args.vocab_size

    print(f"Using vocabulary size: {vocab_size}")

    # Create model configuration based on the user's arguments
    config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False
    }

    # Load data
    docs = load_data(args.data_path, args.max_docs, args.data_format)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(docs, tokenizer, config, args)

    ###########################################################################
    #                            TODO 2.4: YOUR CODE HERE                     #
    #                                                                         #
    # Implement model initialization and setup:                               #
    #                                                                         #
    # 1. Create GPTModel instance with the configuration                      #
    # 2. Move model to the correct device (CPU/GPU)                           #
    # 3. Optionally compile model for better performance                      #
    # 4. Calculate and print parameter counts (optional)                      #
    # 5. Train the model                                                      #
    #                                                                         #
    # NOTE: you probably want mode="default" for the compile mode, but you    #
    #       can experiment with other modes if you want to.                   #
    ###########################################################################

    # 1) Create model
    model = gpt.GPTModel(config)

    # 2) Move to correct device handled inside train_model
    # 3) Optionally compile model
    try:
        model = torch.compile(model, mode="default")  # requires PyTorch 2.x
        print("Model compiled with torch.compile")
    except Exception as e:
        print(f"torch.compile not available or failed ({e}); continuing without compilation.")

    # 4) Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # 5) Train the model
    train_model(model, train_loader, val_loader, config, args)


if __name__ == "__main__":
    main()
