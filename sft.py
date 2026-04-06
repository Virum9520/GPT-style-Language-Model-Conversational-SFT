"""
EECS 595 HW3: Supervised Fine-Tuning (SFT) Implementation

This file contains all the core classes and functions needed to implement
Supervised Fine-Tuning (SFT) of GPT models for conversational AI.

Students should implement the TODO sections in each class and function.
"""

import os
import json
import math
import numpy as np
import random
import logging
from typing import Optional, Callable, List, Tuple, Dict, Any, Iterable
from copy import deepcopy
import gzip

# PyTorch imports
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Transformers and tokenization
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, default_data_collator

# Data handling
from datasets import load_from_disk
import orjson

# Progress tracking
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import wandb

# Import GPT components from gpt.py
from gpt import (
    GPTEmbedding,
    MultiHeadAttention,
    SwiGLU,
    FeedForward,
    RMSNorm as LayerNorm,
    TransformerBlock,
    GPTModel,
    generate_new_tokens,
    generate_text,
    setup_tokenizer
)


# =============================================================================
# SFT Dataset Class
# =============================================================================

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) of GPT models on conversational data.

    Key Features:
    1. Loads conversations from jsonlines format
    2. Formats with special tokens: <|user|>content<|end|><|assistant|>content<|end|>
    3. SELECTIVE MASKING: Only trains on <|assistant|> token and first <|end|> after assistant
    4. Masks all other tokens including <|user|>, <|system|>, and other <|end|> tokens
    """
    def __init__(self, data_file: str, tokenizer, max_length: int = 1024):
        """
        Initialize SFT Dataset.

        Args:
            data_file: Path to jsonlines file with conversations
            tokenizer: Tokenizer with special tokens added
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations: List[List[Dict[str, str]]] = []

        # Precompute special token IDs
        self.SID = {
            "user": tokenizer.convert_tokens_to_ids("<|user|>"),
            "asst": tokenizer.convert_tokens_to_ids("<|assistant|>"),
            "sys": tokenizer.convert_tokens_to_ids("<|system|>"),
            "end": tokenizer.convert_tokens_to_ids("<|end|>"),
            "pad": tokenizer.pad_token_id or 0,
        }

        ofunc = gzip.open if data_file.endswith(".gz") else open
        with ofunc(data_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    self.conversations.append(msgs)

    def __len__(self):
        return len(self.conversations)

    def _build_ids_labels(self, conversation: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build input IDs and labels for a conversation with proper masking.

        Args:
            conversation: List of message dicts

        Returns:
            Tuple of (input_ids, labels) tensors
        """
        ###########################################################################
        #                            TODO 3.1: YOUR CODE HERE                         #
        #                                                                         #
        # Implement fast tokenization with selective masking:                    #
        #                                                                         #
        # 1. Initialize empty lists for ids and labels                           #
        # 2. Iterate through each message in the conversation                     #
        # 3. For each message, extract role and content                           #
        # 4. Based on role, append tokens and labels:                             #
        #    - assistant: Add <|assistant|> token (train), content tokens (train), #
        #                first <|end|> token (train)                               #
        #    - user: Add <|user|> token (mask), content tokens (mask),           #
        #            <|end|> token (mask)                                         #
        #    - system: Add <|system|> token (mask), content tokens (mask),       #
        #              <|end|> token (mask)                                        #
        # 5. Use self.SID dictionary for special token IDs                        #
        # 6. Use tokenizer.encode(text, add_special_tokens=False) for content    #
        # 7. Truncate if sequence exceeds max_length                              #
        # 8. Return as torch tensors                                               #
        #                                                                         #
        ###########################################################################

        ids: List[int] = []
        labs: List[int] = []

        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            content_ids = self.tokenizer.encode(content, add_special_tokens=False)

            # Helper to append with truncation check
            def append_tokens(tok_list: List[int], label_list: List[int]):
                nonlocal ids, labs
                space_left = self.max_length - len(ids)
                if space_left <= 0:
                    return
                take = min(space_left, len(tok_list))
                ids.extend(tok_list[:take])
                labs.extend(label_list[:take])

            if role == "assistant":
                seq_ids = [self.SID["asst"]] + content_ids + [self.SID["end"]]
                seq_labels = seq_ids[:]  # train on assistant tokens including first <|end|>
                append_tokens(seq_ids, seq_labels)
            elif role == "user":
                seq_ids = [self.SID["user"]] + content_ids + [self.SID["end"]]
                seq_labels = [-100] * len(seq_ids)  # mask
                append_tokens(seq_ids, seq_labels)
            elif role == "system":
                seq_ids = [self.SID["sys"]] + content_ids + [self.SID["end"]]
                seq_labels = [-100] * len(seq_ids)  # mask
                append_tokens(seq_ids, seq_labels)

            if len(ids) >= self.max_length:
                break

        return torch.tensor(ids, dtype=torch.long), torch.tensor(labs, dtype=torch.long)

    def __getitem__(self, idx):
        return self._build_ids_labels(self.conversations[idx])


# =============================================================================
# Data Collators
# =============================================================================

def sft_data_collator(batch):
    """
    Custom data collator for SFT dataset that handles tuple format.

    Args:
        batch: List of (input_ids, labels) tuples

    Returns:
        Dictionary with batched input_ids and labels
    """
    ###########################################################################
    #                            TODO 3.2: YOUR CODE HERE                         #
    #                                                                         #
    # Implement SFT data collator for batching:                               #
    #                                                                         #
    # 1. Separate input_ids and labels from batch tuples                      #
    # 2. Find the maximum length in the batch                                #
    # 3. Pad all sequences to the same length:                               #
    #    - Pad input_ids with pad_token_id (usually 0)                       #
    #    - Pad labels with -100 (masked)                                     #
    # 4. Stack into batch tensors                                            #
    # 5. Return dictionary with 'input_ids' and 'labels' keys                #
    #                                                                         #
    # This ensures all sequences in a batch have the same length.            #
    ###########################################################################

    input_seqs, label_seqs = zip(*batch)
    max_len = max(seq.size(0) for seq in input_seqs)
    pad_id = 0

    padded_inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_seqs, label_seqs)):
        L = inp.size(0)
        padded_inputs[i, :L] = inp
        padded_labels[i, :L] = lab

    return {"input_ids": padded_inputs, "labels": padded_labels}


def hf_collate(examples):
    """
    HuggingFace-style collator for packed datasets.

    This collator is designed for pre-packed Arrow datasets where each example
    already contains a full sequence of input_ids and labels. Unlike regular
    datasets that need padding, packed datasets have sequences that are already
    the correct length (max_length).

    The packed format provides several advantages:
    - No padding needed: sequences are already max_length
    - Better GPU utilization: every token contributes to training
    - Faster data loading: Arrow format is optimized for speed
    - Memory efficiency: supports memory mapping for large datasets

    Args:
        examples: List of examples from packed dataset, each containing:
                 - input_ids: List of token IDs (length = max_length)
                 - labels: List of labels with -100 for masked tokens

    Returns:
        Dictionary with batched data:
        - input_ids: Tensor of shape (batch_size, max_length)
        - labels: Tensor of shape (batch_size, max_length)
        - attention_mask: Tensor indicating non-padding tokens
    """
    pad_id = 0  # Default pad token ID
    ids = torch.tensor(np.stack([e["input_ids"] for e in examples]), dtype=torch.long)
    labs = torch.tensor(np.stack([e["labels"] for e in examples]), dtype=torch.long)
    attn = (ids != pad_id).to(torch.long)
    return {"input_ids": ids, "labels": labs, "attention_mask": attn}


# =============================================================================
# Text Generation Functions
# =============================================================================

def _top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    if top_k and top_k > 0:
        kth, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        cutoff = kth[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)
    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        csum = torch.cumsum(probs, dim=-1)
        mask = csum > top_p
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        logits.scatter_(-1, sorted_idx, sorted_logits)
    return logits


def generate_chat_response(model, tokenizer, user_message, max_new_tokens=100, temperature=0.8, context: str = "",
                           top_k=50, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3):
    """
    Generate a conversational response using the fine-tuned model.

    Args:
        model: Fine-tuned GPT model
        tokenizer: Tokenizer with special tokens
        user_message: User's input message
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated response text
    """
    ##############################################################################
    #                            TODO 3.3: YOUR CODE HERE                        #
    #                                                                            #
    # Implement conversational text generation:                                  #
    #                                                                            #
    # 1. Format input with special tokens: "<|user|>message<|end|><|assistant|>" #
    # 2. Tokenize and move to device                                             #
    # 3. Generate tokens autoregressively:                                       #
    #    - Get model output (logits)                                             #
    #    - Apply temperature scaling and sample next token                       #
    #    - Stop if <|end|> token or max length reached                           #
    # 4. Decode and extract assistant's response                                 #
    #                                                                            #
    # This enables conversational AI by generating responses in chat format.     #
    ##############################################################################

    model.eval()
    device = next(model.parameters()).device

    prompt = f"{context}<|user|>{user_message}<|end|><|assistant|>"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")

    with torch.no_grad():
        first_step = True
        for _ in range(max_new_tokens):
            # Respect model context window
            ctx_len = getattr(model, 'context_length', 1024)
            idx_cond = ids[:, -ctx_len:]
            logits = model(idx_cond)[:, -1, :]

            # repetition penalty
            if repetition_penalty and repetition_penalty != 1.0:
                seen = ids[0].unique()
                logits[0, seen] /= repetition_penalty

            # no-repeat n-grams
            if no_repeat_ngram_size and ids.size(1) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                recent = ids[0, -n+1:].tolist()
                if len(recent) == n - 1:
                    for i in range(ids.size(1) - n + 1):
                        if ids[0, i:i + n - 1].tolist() == recent:
                            ban_tok = ids[0, i + n - 1].item()
                            logits[0, ban_tok] = float('-inf')

            # avoid generating <|assistant|> repeatedly after the seed
            if not first_step and assistant_id is not None and assistant_id >= 0:
                logits[:, assistant_id] = float('-inf')

            # temperature + top-k/top-p
            logits = logits / max(1e-6, float(temperature))
            logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
            ids = torch.cat([ids, next_id], dim=1)

            first_step = False
            if end_id is not None and next_id.item() == end_id:
                break

    text = tokenizer.decode(ids[0].tolist())
    if "<|assistant|>" in text:
        after_asst = text.split("<|assistant|>", 1)[1]
        response = after_asst.split("<|end|>", 1)[0]
    else:
        response = text
    return response.strip()

def generate_multi_turn_response(model, tokenizer, conversation_history, max_new_tokens=100, temperature=0.7):
    """
    Generate a response considering the full conversation history.

    Args:
        model: Fine-tuned GPT model
        tokenizer: Tokenizer with special tokens
        conversation_history: List of dicts with 'role' and 'content' keys
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    # Format the conversation history with special tokens (excluding the last user message)
    context = ""
    last_user_message = ""

    for message in conversation_history:
        role = message['role']
        content = message['content']

        if role == 'user':
            context += f"<|user|>{content}<|end|>"
            last_user_message = content  # Keep track of the last user message
        elif role == 'assistant':
            context += f"<|assistant|>{content}<|end|>"
        elif role == 'system':
            context += f"<|system|>{content}<|end|>"

    # Use generate_chat_response with the conversation context
    return generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        user_message=last_user_message,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        context=context
    )


# =============================================================================
# Utility Functions
# =============================================================================

def load_pretrained_model(model_path: str, config: Dict[str, Any]):
    print(f"Loading pre-trained model from {model_path}...")
    ckpt = torch.load(model_path, map_location='cpu')

    # Unwrap common layouts
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt  # already a raw state_dict

    # Strip prefixes like "module." or "_orig_mod."
    keys_head_before = list(state_dict.keys())[:10]
    for prefix in ("module.", "_orig_mod."):
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = { (k[len(prefix):] if k.startswith(prefix) else k): v
                           for k, v in state_dict.items() }
    keys_head_after = list(state_dict.keys())[:10]
    print(f"🏷️ Keys (head) before strip: {keys_head_before}")
    print(f"🏷️ Keys (head) after strip:  {keys_head_after}")

    # Infer vocab size robustly
    emb_key = next((k for k in state_dict if k.endswith("embedding.token_embeddings.weight")), None)
    if emb_key is not None:
        original_vocab_size = state_dict[emb_key].shape[0]
    else:
        out_key = next((k for k in state_dict if k.endswith("out_head.weight")), None)
        if out_key is None:
            raise KeyError("Could not find embedding or out_head weights. Example keys: "
                           + ", ".join(list(state_dict.keys())[:20]))
        original_vocab_size = state_dict[out_key].shape[0]

    new_vocab_size = config['vocab_size']
    if original_vocab_size != new_vocab_size:
        print(f"❌ ERROR: Vocabulary size mismatch!")
        print(f"   Model vocab size: {original_vocab_size}")
        print(f"   Expected vocab size: {new_vocab_size}")
        raise ValueError("Vocabulary size mismatch - use the corrected model")

    model = GPTModel(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"🔎 Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"   Missing (head): {missing[:10]}")
    if unexpected:
        print(f"   Unexpected (head): {unexpected[:10]}")
    print(f"✅ Model loaded successfully!")
    print(f"✅ Model vocabulary size: {model.embedding.token_embeddings.weight.shape[0]}")
    try:
        model.eval()
        with torch.no_grad():
            test_inp = torch.randint(0, new_vocab_size, (1, 8))
            test_out = model(test_inp)
            print(f"✅ Sanity forward OK: output shape {tuple(test_out.shape)}")
    except Exception as e:
        print(f"⚠️ Sanity forward failed: {e}")
    return model


def evaluate_validation_loss(model, val_loader, loss_fn, device, max_batches: int | None = None):
    """
    Evaluate the model's loss on the validation dataset.

    Args:
        model: The GPT model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run evaluation on

    Returns:
        Average validation loss
    """
    ###########################################################################
    #                            TODO 3.4: YOUR CODE HERE                     #
    #                                                                         #
    # Implement validation loss evaluation for SFT:                           #
    #                                                                         #
    # 1. Set model to evaluation mode (model.eval())                          #
    # 2. Initialize loss tracking variables                                   #
    # 3. Iterate through validation batches with torch.no_grad():             #
    #    - Handle different batch formats (dict vs tuple)                     #
    #    - Move data to device                                                #
    #    - Forward pass with mixed precision (optional)                       #
    #    - Compute loss and accumulate                                        #
    # 4. Calculate average validation loss                                    #
    # 5. Set model back to training mode (model.train())                      #
    # 6. Return the average validation loss                                   #
    #                                                                         #
    # This is crucial for monitoring SFT training progress!                   #
    ###########################################################################

    model.eval()
    total_loss = 0.0
    count = 0
    device_type = 'cuda' if str(device).startswith('cuda') else ('mps' if str(device) == 'mps' else 'cpu')
    amp_dtype = torch.bfloat16 if device_type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
            else:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

            with autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type != 'cpu')):
                logits = model(input_ids)
                logits = logits[:, :-1, :].contiguous()
                targets = labels[:, 1:].contiguous()
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()
            count += 1

    avg = total_loss / max(1, count)
    model.train()
    return avg


def create_sft_dataloader(data_file: str, tokenizer, batch_size: int = 16,
                         max_length: int = 1024, shuffle: bool = True,
                         drop_last: bool = True, num_workers: int = 0,
                         use_packed: bool = False):
    """
    Create a DataLoader for SFT training.

    This function supports two data formats:
    1. **Regular format** (use_packed=False): Individual conversations in jsonlines format
    2. **Packed format** (use_packed=True): Pre-processed Arrow dataset with packed sequences

    Packed datasets are more efficient for training because they:
    - Maximize GPU utilization by filling sequences to max_length
    - Reduce padding overhead
    - Enable faster data loading with Arrow format
    - Support memory mapping for large datasets

    Args:
        data_file: Path to data file (jsonlines or packed dataset)
        tokenizer: Tokenizer with special tokens
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes
        use_packed: Whether to use packed dataset format

    Returns:
        DataLoader instance
    """
    ###########################################################################
    #                            TODO 3.5: YOUR CODE HERE                     #
    #                                                                         #
    # Implement SFT DataLoader creation:                                      #
    #                                                                         #
    # 1. Check if use_packed is True or False                                 #
    # 2. If use_packed=True:                                                  #
    #    - Use load_from_disk() to load packed dataset                        #
    #    - Create DataLoader with hf_collate function                         #
    # 3. If use_packed=False:                                                 #
    #    - Create SFTDataset instance with data_file, tokenizer, max_length   #
    #    - Create DataLoader with sft_data_collator function                  #
    # 4. Print success message and return DataLoader                          #
    #                                                                         #
    # This provides flexibility to use either regular or packed datasets.     #
    ###########################################################################

    if use_packed:
        dataset = load_from_disk(data_file)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers, collate_fn=hf_collate)
        print(f"✅ Loaded packed dataset from {data_file}")
        return loader
    else:
        dataset = SFTDataset(data_file=data_file, tokenizer=tokenizer, max_length=max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers, collate_fn=sft_data_collator)
        print(f"✅ Loaded regular SFT dataset from {data_file}")
        return loader
