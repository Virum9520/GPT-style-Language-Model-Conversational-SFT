# EECS 595 HW3: GPT Implementation

Welcome to the GPT homework! This directory contains a modular implementation of a GPT-style language model that you'll build step by step. The code is organized into three main files to help you learn each component systematically.

## 📁 File Structure

### 1. `gpt.py` - Your Implementation File
**What you'll do**: Implement all the TODO sections marked in this file.

**What's inside**:
- **Core GPT Components**: All the building blocks you need to implement
- **RoPE Integration**: Modern positional encoding using Rotary Position Embeddings
- **SwiGLU Activation**: Advanced activation function for better performance
- **Dataset Classes**: Data loading utilities for training
- **Text Generation**: Functions to generate text from your trained model

**Key Classes to Implement**:
- `GPTEmbedding` - Converts token IDs to embeddings (no positional embeddings needed!)
- `MultiHeadAttention` - The attention mechanism with RoPE built-in
- `SwiGLU` - Modern activation function
- `FeedForward` - Position-wise MLP layer
- `TransformerBlock` - Combines attention and MLP with residual connections
- `GPTModel` - The complete GPT model
- `GPTDataset` - Handles data loading for training

### 2. `Debug_GPT_Pretraining.ipynb` - Your Testing Notebook
**What you'll do**: Run this notebook to verify each component works correctly.

**How it helps**:
- **Step-by-step verification**: Test each component as you implement it
- **Automatic reloading**: Uses `importlib.reload()` so you can test changes immediately
- **Clear feedback**: Shows ✅ for success and ⚠️ for warnings
- **Educational explanations**: Understand what each test is checking

**Testing Order**:
1. Tokenizer setup
2. GPTEmbedding layer
3. MultiHeadAttention (with RoPE)
4. SwiGLU activation
5. FeedForward layer
6. TransformerBlock
7. Complete GPTModel
8. Text generation
9. Dataset creation
10. DataLoader testing

### 3. `pretrain_gpt.py` - Training Script
**What you'll do**: Run this to train your implemented model.

**Features**:
- **Mixed precision training**: Faster training with FP16
- **Wandb logging**: Track your training progress
- **Model checkpointing**: Save models during training
- **Validation evaluation**: Monitor overfitting
- **Flexible configuration**: Easy to adjust hyperparameters

### 4. `test_gpt.py` - Unit Tests
**What you'll do**: Run this to verify your implementation works correctly.

**Features**:
- **Comprehensive testing**: Tests all major components
- **Shape verification**: Ensures correct tensor dimensions
- **Functional testing**: Verifies components work as expected
- **Integration testing**: Tests complete pipeline
- **Deterministic testing**: Ensures reproducible results

**Test Coverage**:
- `GPTEmbedding`: Token embedding functionality
- `MultiHeadAttention`: Attention mechanism with RoPE
- `SwiGLU`: Activation function
- `FeedForward`: MLP layer
- `LayerNorm`: Normalization layer
- `TransformerBlock`: Complete transformer block
- `GPTModel`: Full model
- `GPTDataset`: Data loading
- Generation functions: Text generation
- Utility functions: Tokenizer and DataLoader creation

### 5. `sft.py` - Supervised Fine-Tuning Implementation
**What you'll do**: Implement SFT components for conversational AI.

**Features**:
- **SFTDataset**: Load and format conversational data with proper token masking
- **SFTDatasetFast**: Fast tokenization version for better performance
- **Data collators**: Handle batching for SFT training
- **Generation functions**: Conversational text generation
- **Utility functions**: Model loading and validation

**Key Components**:
- `SFTDataset`: Loads conversations from jsonlines format
- `SFTDatasetFast`: Efficient tokenization per-message
- `sft_data_collator`: Custom batching for SFT data
- `generate_chat_response`: Single-turn conversation generation
- `generate_multi_turn_response`: Multi-turn conversation generation
- `evaluate_validation_loss`: Validation loss computation

### 6. `Debug_SFT_Training.ipynb` - SFT Debug Notebook
**What you'll do**: Interactive testing of SFT components.

**Features**:
- **Step-by-step verification**: Test each SFT component individually
- **Conversation formatting**: Verify special token handling
- **Token masking**: Test selective training logic
- **Generation testing**: Test conversational inference
- **Integration testing**: Complete SFT pipeline verification

### 7. `sft_gpt.py` - SFT Training Script
**What you'll do**: Run this to fine-tune your pre-trained model for conversations.

**Features**:
- **Pre-trained model loading**: Load your trained GPT model
- **SFT training loop**: Fine-tune with masked loss computation
- **Mixed precision training**: Faster training with FP16
- **Wandb logging**: Track your SFT training progress
- **Model checkpointing**: Save models during SFT training
- **Validation evaluation**: Monitor SFT performance

### 8. Training Scripts
- **`sft_full.sh`**: Full-scale SFT training on cluster
- **`sft_tiny.sh`**: Local SFT verification with tiny dataset

## 🚀 Getting Started

### Step 1: Implement Core Components (Recommended Order)

Follow this TODO order for the most logical progression from simple to complex:

#### Phase 1: Foundation Components (Start Here)
1. **TODO 1.1 & 1.2**: `GPTEmbedding` - Token embeddings only (no positional embeddings needed)
2. **TODO 1.5 & 1.6**: `FeedForward` - MLP layer with SwiGLU activation
3. **TODO 1.15**: `setup_tokenizer` - Tokenizer setup with special tokens

#### Phase 2: Attention Mechanism (Most Complex)
4. **TODO 1.3 & 1.4**: `MultiHeadAttention` - Attention with RoPE (most challenging component)

#### Phase 3: Transformer Architecture
5. **TODO 1.7, 1.8 & 1.9**: `TransformerBlock` - Combines attention and MLP with residual connections
6. **TODO 1.10 & 1.11**: `GPTModel` - Complete GPT model assembly

#### Phase 4: Data Handling
7. **TODO 1.12 & 1.13**: `GPTDataset` - Data loading for pretraining
8. **TODO 1.14**: `create_dataloader` - DataLoader creation utility

#### Phase 5: Pretraining Script TODOs (After components are implemented)

9. **TODO 2.1**: `create_dataloaders` - Training/validation dataloader creation
10. **TODO 2.2**: `evaluate_validation_loss` - Validation evaluation
11. **TODO 2.3**: Training loop gradient accumulation and optimization
12. **TODO 2.4**: Model initialization and setup

Once you have these done, you can try training your model!

#### Phase 5: SFT Components (After GPT is working)
13. **TODO 3.1**: `SFTDataset._build_ids_labels` - Conversation formatting with token masking
14. **TODO 3.2**: `sft_data_collator` - Custom batching for SFT data
15. **TODO 3.3**: `generate_chat_response` - Single-turn conversation generation
16. **TODO 3.4**: `evaluate_validation_loss` - Validation loss computation
17. **TODO 3.5**: `create_sft_dataloader` - SFT DataLoader creation

#### Phase 6: SFT Script TODOs (After components are implemented)

18. **TODO 4.1**: SFT dataloader creation
19. **TODO 4.2**: SFT forward pass implementation
20. **TODO 4.3**: SFT backward pass and optimization
21. **TODO 4.4**: SFT model loading and setup

### Step 2: Test Your Implementation
```bash
# Run comprehensive unit tests
python test_gpt.py

# Run tests with verbose output
python test_gpt.py -v

# Run specific test class
python -m unittest test_gpt.TestGPTEmbedding -v
```

### Step 3: Interactive Testing
```bash
# Open the debug notebook
jupyter notebook Debug_GPT_Pretraining.ipynb
```

Run each cell in order. If a test fails, go back to `gpt.py` and fix the implementation.

### Step 4: Train Your Model

#### For Local Testing (Small Model)
```bash
# This runs a tiny model that trains quickly on CPU
./train_tiny.sh
```

#### For Full Training (Production Model)
```bash
# This runs the full model (requires GPU)
./train_full.sh
```

#### Manual Training
```bash
# Basic training
python pretrain_gpt.py

# Custom parameters
python pretrain_gpt.py \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --max_epochs 3 \
    --emb_dim 256 \
    --n_layers 6
```

## 🧪 Testing Strategy

### Unit Tests (`test_gpt.py`)
Run comprehensive unit tests to verify each component:

```bash
# Run all tests
python test_gpt.py

# Run with verbose output
python test_gpt.py -v

# Run specific test classes
python -m unittest test_gpt.TestGPTEmbedding -v
python -m unittest test_gpt.TestMultiHeadAttention -v
python -m unittest test_gpt.TestSwiGLU -v
python -m unittest test_gpt.TestFeedForward -v
python -m unittest test_gpt.TestLayerNorm -v
python -m unittest test_gpt.TestTransformerBlock -v
python -m unittest test_gpt.TestGPTModel -v
python -m unittest test_gpt.TestGPTDataset -v
python -m unittest test_gpt.TestIntegration -v
```

### Interactive Testing (`Debug_GPT_Pretraining.ipynb`)
Use the debug notebook for step-by-step verification:

1. **Cell-by-cell testing**: Test each component individually
2. **Immediate feedback**: See results as you implement
3. **Educational explanations**: Understand what each test verifies
4. **Reload functionality**: Use `importlib.reload()` to test changes

### Testing Order
1. **Unit tests first**: Run `test_gpt.py` to catch basic issues
2. **Interactive testing**: Use the debug notebook for detailed verification
3. **Integration testing**: Test the complete pipeline
4. **Training verification**: Run tiny training to ensure everything works

## 🎯 Implementation Plan

### Phase 1: Core GPT Implementation (Weeks 1-2)
1. **Start with basic components**:
   - **TODO 1.1 & 1.2**: `GPTEmbedding` - Token embeddings only (no positional embeddings)
   - **TODO 1.5 & 1.6**: `FeedForward` - MLP layer with SwiGLU activation
   - **TODO 1.15**: `setup_tokenizer` - Tokenizer setup with special tokens

2. **Implement attention mechanism**:
   - **TODO 1.3 & 1.4**: `MultiHeadAttention` - Most complex component with RoPE
   - Study RoPE implementation carefully
   - Test positional sensitivity

3. **Build complete model**:
   - **TODO 1.7, 1.8 & 1.9**: `TransformerBlock` - Combines attention and MLP
   - **TODO 1.10 & 1.11**: `GPTModel` - Complete GPT architecture
   - Test with different configurations

4. **Data handling**:
   - **TODO 1.12 & 1.13**: `GPTDataset` - Data loading for pretraining
   - **TODO 1.14**: `create_dataloader` - DataLoader creation
   - Test with sample data

### Phase 2: Testing and Verification (Week 2-3)
1. **Unit testing**:
   ```bash
   python test_gpt.py
   ```

2. **Interactive testing**:
   ```bash
   jupyter notebook Debug_GPT_Pretraining.ipynb
   ```

3. **Training verification**:
   ```bash
   ./train_tiny.sh  # Local verification
   ```

### Phase 3: SFT Implementation (Week 3-4)
1. **SFT data handling**:
   - **TODO 3.1**: `SFTDataset._build_ids_labels` - Load conversations from jsonlines with token masking
   - **TODO 3.5**: `create_sft_dataloader` - DataLoader creation for SFT

2. **Data collation**:
   - **TODO 3.2**: `sft_data_collator` - Handle batching for SFT data

3. **Generation functions**:
   - **TODO 3.3**: `generate_chat_response` - Single-turn conversations
   - **TODO 3.4**: `evaluate_validation_loss` - Validation loss computation

4. **SFT testing**:
   ```bash
   jupyter notebook Debug_SFT_Training.ipynb
   ```

### Phase 4: Training and Evaluation (Week 4-5)
1. **Pretraining script implementation**:
   - **TODO 2.1**: `create_dataloaders` - Training/validation dataloader creation
   - **TODO 2.2**: `evaluate_validation_loss` - Validation evaluation
   - **TODO 2.3**: Training loop gradient accumulation and optimization
   - **TODO 2.4**: Model initialization and setup

2. **SFT training script implementation**:
   - **TODO 4.1**: SFT dataloader creation
   - **TODO 4.2**: SFT forward pass implementation
   - **TODO 4.3**: SFT backward pass and optimization
   - **TODO 4.4**: SFT model loading and setup

3. **Training execution**:
   ```bash
   ./train_full.sh  # Full model training
   ./sft_full.sh    # Fine-tune for conversations
   ```

4. **Evaluation**:
   - Test conversational generation
   - Evaluate model performance
   - Compare before/after SFT

## 📋 Complete TODO Reference

### GPT Core Components (`gpt.py`)
- **TODO 1.1**: `GPTEmbedding.__init__` - Initialize token embedding layer
- **TODO 1.2**: `GPTEmbedding.forward` - Forward pass for token embeddings
- **TODO 1.3**: `MultiHeadAttention.__init__` - Initialize attention with RoPE
- **TODO 1.4**: `MultiHeadAttention.forward` - Forward pass with RoPE attention
- **TODO 1.5**: `FeedForward.__init__` - Initialize MLP with SwiGLU
- **TODO 1.6**: `FeedForward.forward` - Forward pass through MLP
- **TODO 1.7**: `TransformerBlock.__init__` - Initialize transformer block
- **TODO 1.8**: `TransformerBlock.maybe_dropout` - Apply dropout conditionally
- **TODO 1.9**: `TransformerBlock.forward` - Forward pass through transformer block
- **TODO 1.10**: `GPTModel.__init__` - Initialize complete GPT model
- **TODO 1.11**: `GPTModel.forward` - Forward pass through GPT model
- **TODO 1.12**: `GPTDataset.__init__` - Initialize dataset for pretraining
- **TODO 1.13**: `GPTDataset.__getitem__` - Get dataset samples
- **TODO 1.14**: `create_dataloader` - Create DataLoader for training
- **TODO 1.15**: `setup_tokenizer` - Setup tokenizer with special tokens

### Pretraining Script (`pretrain_gpt.py`)
- **TODO 2.1**: `create_dataloaders` - Create train/validation dataloaders
- **TODO 2.2**: `evaluate_validation_loss` - Evaluate model on validation set
- **TODO 2.3**: Training loop gradient accumulation and optimization
- **TODO 2.4**: Model initialization and setup

### SFT Components (`sft.py`)
- **TODO 3.1**: `SFTDataset._build_ids_labels` - Format conversations with token masking
- **TODO 3.2**: `sft_data_collator` - Custom data collator for SFT batching
- **TODO 3.3**: `generate_chat_response` - Single-turn conversation generation
- **TODO 3.4**: `evaluate_validation_loss` - Validation loss evaluation for SFT
- **TODO 3.5**: `create_sft_dataloader` - Create DataLoader for SFT training

### SFT Training Script (`sft_gpt.py`)
- **TODO 4.1**: `create_dataloaders` - Create SFT train/validation dataloaders
- **TODO 4.2**: SFT forward pass implementation
- **TODO 4.3**: SFT backward pass and optimization
- **TODO 4.4**: SFT model loading and setup

## 📚 Key Concepts to Understand

### GPT Architecture
- **Transformer blocks**: Self-attention + MLP with residual connections
- **Causal masking**: Prevent attention to future tokens
- **Next-token prediction**: Standard language modeling objective

### RoPE (Rotary Position Embedding)
- **Positional encoding**: Encodes position directly in attention
- **Relative positions**: Better handling of sequence length
- **Implementation**: Rotate queries and keys based on position

### SFT (Supervised Fine-Tuning)
- **Conversation format**: Special tokens for user/assistant/system
- **Selective masking**: Only train on assistant responses
- **Context preservation**: User messages provide context but don't contribute to loss
- **Arrow datasets**: Efficient data format for large-scale training
- **Packing**: Optimize sequence utilization by combining multiple conversations

### Training Techniques
- **Mixed precision**: FP16/bfloat16 for faster training
- **Gradient accumulation**: Larger effective batch sizes
- **Learning rate scheduling**: Cosine schedule with warmup
- **Gradient clipping**: Prevent exploding gradients

### Arrow Datasets

#### What are Arrow Datasets?
[Arrow datasets](https://huggingface.co/docs/datasets/about_arrow) are a high-performance data format designed for efficient data processing and machine learning workflows. They provide:

- **Columnar storage**: Data is stored in columns rather than rows, enabling efficient compression and fast access
- **Memory mapping**: Large datasets can be accessed without loading entirely into memory
- **Cross-language compatibility**: Arrow format works across Python, R, Java, C++, etc.
- **Lazy evaluation**: Operations are computed on-demand, reducing memory usage
- **Parallel processing**: Built-in support for multi-core data processing

We've provided already-prepared Arrow datasets for both training and SFT on Canvas, in addition to the original data. For reference, we've also inlcuded a `convert_to_arrow.ipynb` notebook so you see how this process worked, but you don't need to understand it.

#### Why Use Arrow Datasets for Pretraining and SFT?
1. **Memory efficiency**: Large conversation datasets don't need to fit entirely in RAM (Needed for Great Lakes!)
2. **Fast loading**: Arrow format is optimized for quick data access
3. **Batch processing**: Efficient handling of large batches during training
4. **Caching**: Arrow datasets can be cached and reused across training runs
5. **Streaming**: Support for streaming large datasets that don't fit in memory

Since we've already prepared and packaged the Arrow datasets, they are also much easier to use in your code and the implementations for data loading and collating will be easier.

#### Data Loading Strategy in This Assignment

The Pretraining and SFT implementations uses a hybrid approach:

- **Training**: Packed Arrow dataset for efficiency
- **Validation**: Regular dataset for easier debugging and evaluation

This gives you the best of both worlds: efficient training with packed data and clear validation with individual conversations. You can get started using the non-Arrow datasets for both (they're included in Canvas). However, when training, you will _absolutely want to use the Arrow datasets_ that we've prepared for you. These will give a much-needed performance boost and keep the RAM requirements for your job much lower.

### Component-Specific Tips

#### GPTEmbedding
- Only implement token embeddings (no positional embeddings)
- RoPE handles positional information in the attention layer
- Use `nn.Embedding` for token embeddings

#### MultiHeadAttention
- This is the most complex component
- RoPE is applied to queries and keys before attention computation
- Use the provided `apply_rotary_pos_emb` function
- Don't forget causal masking for GPT

#### SwiGLU
- Implement the gating mechanism: `main * Swish(gate)`
- Use separate linear layers for main and gate paths
- Swish is `x * sigmoid(x)`

#### FeedForward
- Use SwiGLU as the activation function
- Typical expansion factor is 4x the embedding dimension
- Return to original embedding dimension

#### SFT-Specific Components

##### SFTDataset
- **Conversation formatting**: Add special tokens `<|user|>`, `<|assistant|>`, `<|end|>`
- **Selective masking**: Only train on assistant tokens (labels != -100)
- **Context preservation**: User/system messages provide context but are masked

##### Token Masking Logic
- **Assistant tokens**: Train on `<|assistant|>` token and content
- **User tokens**: Mask `<|user|>` token and content (labels = -100)
- **System tokens**: Mask `<|system|>` token and content (labels = -100)
- **End tokens**: Train only on first `<|end|>` after assistant content

##### Data Collation
- **Padding**: Pad sequences to same length in batch
- **Masking**: Pad labels with -100 (ignored by loss)
- **Efficiency**: Use appropriate collator for dataset format

##### Generation Functions
- **Format input**: Use proper conversation format with special tokens
- **Autoregressive generation**: Sample tokens one by one
- **Stop conditions**: Stop at `<|end|>` token or max length
- **Response extraction**: Extract only assistant's response

### Testing Strategy
1. **Shape verification**: Always check output tensor shapes
2. **NaN detection**: Make sure no NaN values in outputs
3. **Functional tests**: Verify components behave as expected
4. **Integration tests**: Ensure components work together

## 🔧 Key Differences from Traditional GPT

### RoPE vs Positional Embeddings
- **Traditional GPT**: Token embeddings + positional embeddings
- **This implementation**: Token embeddings only + RoPE in attention
- **Benefits**: Better handling of longer sequences, more efficient

### Modern Components
- **SwiGLU activation**: Better than ReLU/GELU
- **RMSNorm**: Smoother normalization
- **Mixed precision training**: Faster and more memory efficient
- **Gradient accumulation**: Train with larger effective batch sizes

## 📊 Training Scripts

### `train_tiny.sh` - Local Testing
- **Purpose**: Verify your implementation works
- **Model size**: Very small (32 dims, 2 layers)
- **Training time**: ~5-10 minutes on CPU
- **Use case**: Quick verification before full training

### `train_full.sh` - Production Training
- **Purpose**: Train the complete model
- **Model size**: Full size (512 dims, 12 layers)
- **Training time**: Several hours on GPU
- **Use case**: Final model for evaluation

## 🐛 Troubleshooting

### Common Issues
1. **Shape mismatches**: Check tensor dimensions at each step
2. **NaN values**: Usually indicates division by zero or overflow
3. **Import errors**: Make sure all dependencies are installed
4. **CUDA errors**: Check if GPU is available and properly configured

### Debugging Tips
- Use the debug notebook to isolate issues
- Print tensor shapes at each step
- Check for NaN values with `torch.isnan()`
- Use smaller batch sizes for debugging

## 📈 Success Criteria

Your implementation is successful when:
- ✅ All debug notebook tests pass
- ✅ Model trains without errors
- ✅ Generated text shows some coherence (even if random)
- ✅ Training loss decreases over time
- ✅ No NaN values in training

## 🔄 Complete Workflow

### 1. Pretraining Workflow
```bash
# Implement core GPT components
python test_gpt.py                    # Unit tests
jupyter notebook Debug_GPT_Pretraining.ipynb  # Interactive testing

# Train the model
./train_tiny.sh                      # Local verification
./train_full.sh                      # Full training
```

### 2. SFT Workflow
```bash
# Implement SFT components
jupyter notebook Debug_SFT_Training.ipynb  # SFT testing

# Fine-tune for conversations
./sft_tiny.sh                        # Local SFT verification
./sft_full.sh                        # Full SFT training
```

### 3. Evaluation Workflow
```bash
# Test conversational generation
python -c "
import sft, gpt, torch
# Load your trained model
# Test generate_chat_response()
# Test generate_multi_turn_response()
"
```

## 🎓 Learning Objectives

By completing this homework, you'll understand:
- **GPT Architecture**: Transformer blocks, attention mechanisms, causal masking
- **RoPE**: Modern positional encoding techniques
- **SFT**: Supervised fine-tuning for conversational AI
- **Token Masking**: Selective training strategies
- **Mixed Precision Training**: FP16/bfloat16 optimization
- **Data Loading**: Efficient data pipelines for language models
- **Training Loops**: Complete training and fine-tuning workflows
- **Text Generation**: Autoregressive generation techniques

## 🏆 Success Criteria

Your implementation is successful when:
- ✅ All unit tests pass (`test_gpt.py`)
- ✅ All debug notebook tests pass
- ✅ Model trains without errors (pretraining)
- ✅ SFT training completes successfully
- ✅ Generated text shows conversational patterns
- ✅ Model responds appropriately to user messages
- ✅ No NaN values in training
- ✅ Training losses decrease over time

## 🚨 Common Pitfalls

### GPT Implementation
- **Shape mismatches**: Check tensor dimensions at each step
- **RoPE integration**: Ensure RoPE is applied to queries and keys
- **Causal masking**: Don't forget upper-triangular mask
- **Residual connections**: Add input to output in transformer blocks

### SFT Implementation
- **Token masking**: Only train on assistant generated tokens (labels != -100)
- **Special tokens**: Ensure all special tokens are properly handled
- **Conversation format**: Use correct format with `<|user|>`, `<|assistant|>`, `<|end|>`
- **Data collation**: Handle padding and masking correctly

### Training Issues
- **Gradient explosion**: Use gradient clipping
- **Memory issues**: Reduce batch size or use gradient accumulation
- **Slow training**: Enable mixed precision
- **Convergence**: Check learning rate and warmup

Good luck with your implementation! Remember to test each component thoroughly before moving to the next one.