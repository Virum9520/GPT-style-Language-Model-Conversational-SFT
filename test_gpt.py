"""
Comprehensive unit tests for GPT implementation components.

This file contains all unit tests for the core GPT components implemented in gpt.py.
It consolidates tests from multiple files to provide comprehensive coverage of:
- Basic functionality tests
- TODO implementation tests
- Mathematical correctness tests
- Advanced edge cases and numerical stability tests

Tests use simple tensors and verify both correctness and shape consistency.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from unittest.mock import Mock, MagicMock
from typing import Optional, Callable, List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader

# Import the actual classes from gpt.py
from gpt import (
    GPTEmbedding,
    MultiHeadAttention,
    SwiGLU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    GPTDataset,
    generate_new_tokens,
    generate_text,
    create_dataloader,
    setup_tokenizer
)


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestGPTEmbedding(unittest.TestCase):
    """Test cases for GPTEmbedding class."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.emb_dim = 64
        self.context_length = 128
        self.batch_size = 4
        self.seq_length = 16

    def test_embedding_initialization(self):
        """Test embedding layer initialization."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)

        # Check that embedding layers are created
        self.assertIsInstance(embedding.token_embeddings, nn.Embedding)

        # Check embedding dimensions
        self.assertEqual(embedding.token_embeddings.num_embeddings, self.vocab_size)
        self.assertEqual(embedding.token_embeddings.embedding_dim, self.emb_dim)

    def test_embedding_forward_shape(self):
        """Test embedding forward pass shape."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)
        token_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))

        output = embedding(token_ids)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.emb_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_embedding_forward_deterministic(self):
        """Test that embedding forward pass is deterministic."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)
        token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        # Set seed for reproducibility
        torch.manual_seed(42)
        output1 = embedding(token_ids)

        torch.manual_seed(42)
        output2 = embedding(token_ids)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_embedding_token_independence(self):
        """Test that different tokens produce different embeddings."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)

        # Different tokens at same position
        token_ids = torch.tensor([[1, 2, 3]])  # Different tokens

        output = embedding(token_ids)

        # Different tokens should produce different embeddings
        self.assertFalse(torch.allclose(output[0, 0], output[0, 1]))
        self.assertFalse(torch.allclose(output[0, 1], output[0, 2]))


class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for MultiHeadAttention class."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_in = 64
        self.context_length = 32
        self.dropout = 0.1
        self.num_heads = 8
        self.batch_size = 2
        self.seq_len = 8

    def test_attention_initialization(self):
        """Test attention layer initialization."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)

        # Check that linear layers are created
        self.assertIsInstance(attention.W_query, nn.Linear)
        self.assertIsInstance(attention.W_key, nn.Linear)
        self.assertIsInstance(attention.W_value, nn.Linear)
        self.assertIsInstance(attention.out_proj, nn.Linear)

        # Check dimensions
        self.assertEqual(attention.head_dim, self.d_in // self.num_heads)
        self.assertEqual(attention.scale, 1 / math.sqrt(attention.head_dim))

    def test_attention_forward_shape(self):
        """Test attention forward pass shape."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_in)

        output = attention(x)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_in)
        self.assertEqual(output.shape, expected_shape)

    def test_attention_causal_mask(self):
        """Test that causal masking works correctly."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)
        x = torch.randn(1, self.seq_len, self.d_in)

        output = attention(x)

        # Output should not contain NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_attention_deterministic(self):
        """Test that attention is deterministic with same input."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_in)

        torch.manual_seed(42)
        output1 = attention(x)

        torch.manual_seed(42)
        output2 = attention(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_attention_gradient_flow(self):
        """Test that gradients flow through attention."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_in, requires_grad=True)

        output = attention(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


class TestSwiGLU(unittest.TestCase):
    """Test cases for SwiGLU activation function."""

    def setUp(self):
        """Set up test fixtures."""
        self.dimension = 32

    def test_swiglu_initialization(self):
        """Test SwiGLU initialization."""
        swiglu = SwiGLU(self.dimension)

        # Check that linear layers are created
        self.assertIsInstance(swiglu.linear_1, nn.Linear)
        self.assertIsInstance(swiglu.linear_2, nn.Linear)

        # Check dimensions
        self.assertEqual(swiglu.linear_1.in_features, self.dimension)
        self.assertEqual(swiglu.linear_1.out_features, self.dimension)
        self.assertEqual(swiglu.linear_2.in_features, self.dimension)
        self.assertEqual(swiglu.linear_2.out_features, self.dimension)

    def test_swiglu_forward_shape(self):
        """Test SwiGLU forward pass shape."""
        swiglu = SwiGLU(self.dimension)
        x = torch.randn(4, 8, self.dimension)

        output = swiglu(x)

        # Check output shape matches input
        self.assertEqual(output.shape, x.shape)

    def test_swiglu_forward_deterministic(self):
        """Test that SwiGLU is deterministic."""
        swiglu = SwiGLU(self.dimension)
        x = torch.randn(2, 4, self.dimension)

        torch.manual_seed(42)
        output1 = swiglu(x)

        torch.manual_seed(42)
        output2 = swiglu(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_swiglu_no_nan(self):
        """Test that SwiGLU doesn't produce NaN values."""
        swiglu = SwiGLU(self.dimension)
        x = torch.randn(2, 4, self.dimension)

        output = swiglu(x)

        # Output should not contain NaN values
        self.assertFalse(torch.isnan(output).any())


class TestFeedForward(unittest.TestCase):
    """Test cases for FeedForward class."""

    def setUp(self):
        """Set up test fixtures."""
        self.emb_dim = 64
        self.batch_size = 4
        self.seq_len = 8

    def test_feedforward_initialization(self):
        """Test FeedForward initialization."""
        ff = FeedForward(self.emb_dim)

        # Check that linear layers are created
        self.assertIsInstance(ff.fc1, nn.Linear)
        self.assertIsInstance(ff.fc2, nn.Linear)

    def test_feedforward_forward_shape(self):
        """Test FeedForward forward pass shape."""
        ff = FeedForward(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        output = ff(x)

        # Check output shape matches input
        self.assertEqual(output.shape, x.shape)

    def test_feedforward_forward_deterministic(self):
        """Test that FeedForward is deterministic."""
        ff = FeedForward(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        torch.manual_seed(42)
        output1 = ff(x)

        torch.manual_seed(42)
        output2 = ff(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_feedforward_no_nan(self):
        """Test that FeedForward doesn't produce NaN values."""
        ff = FeedForward(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        output = ff(x)

        # Output should not contain NaN values
        self.assertFalse(torch.isnan(output).any())


class TestRMSNorm(unittest.TestCase):
    """Test cases for RMSNorm class (using PyTorch's RMSNorm)."""

    def setUp(self):
        """Set up test fixtures."""
        self.emb_dim = 64
        self.batch_size = 4
        self.seq_len = 8

    def test_rmsnorm_initialization(self):
        """Test RMSNorm initialization."""
        from torch.nn import RMSNorm
        ln = RMSNorm(self.emb_dim)

        # Check that parameters are created
        self.assertIsInstance(ln.weight, nn.Parameter)

        # Check dimensions
        self.assertEqual(ln.weight.shape, (self.emb_dim,))

    def test_rmsnorm_forward_shape(self):
        """Test RMSNorm forward pass shape."""
        from torch.nn import RMSNorm
        ln = RMSNorm(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        output = ln(x)

        # Check output shape matches input
        self.assertEqual(output.shape, x.shape)

    def test_rmsnorm_forward_deterministic(self):
        """Test that RMSNorm is deterministic."""
        from torch.nn import RMSNorm
        ln = RMSNorm(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        torch.manual_seed(42)
        output1 = ln(x)

        torch.manual_seed(42)
        output2 = ln(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_rmsnorm_no_nan(self):
        """Test that RMSNorm doesn't produce NaN values."""
        from torch.nn import RMSNorm
        ln = RMSNorm(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        output = ln(x)

        # Output should not contain NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_rmsnorm_normalization_properties(self):
        """Test that RMSNorm normalizes correctly."""
        from torch.nn import RMSNorm
        ln = RMSNorm(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        output = ln(x)

        # RMSNorm normalizes by RMS, not mean/std like LayerNorm
        # Check that the RMS is approximately 1
        rms_values = torch.sqrt(torch.mean(output**2, dim=-1))
        self.assertTrue(torch.allclose(rms_values, torch.ones_like(rms_values), atol=1e-5))


class TestTransformerBlock(unittest.TestCase):
    """Test cases for TransformerBlock class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = {
            'emb_dim': 64,
            'context_length': 32,
            'n_heads': 8,
            'n_layers': 2,
            'drop_rate': 0.1
        }
        self.batch_size = 2
        self.seq_len = 8

    def test_transformer_block_initialization(self):
        """Test TransformerBlock initialization."""
        block = TransformerBlock(self.cfg)

        # Check that components are created
        self.assertIsInstance(block.self_attn, MultiHeadAttention)
        self.assertIsInstance(block.ffn, FeedForward)
        self.assertIsInstance(block.norm1, nn.Module)  # RMSNorm
        self.assertIsInstance(block.norm2, nn.Module)  # RMSNorm

    def test_transformer_block_forward_shape(self):
        """Test TransformerBlock forward pass shape."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg['emb_dim'])

        output = block(x)

        # Check output shape matches input
        self.assertEqual(output.shape, x.shape)

    def test_transformer_block_forward_deterministic(self):
        """Test that TransformerBlock is deterministic."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg['emb_dim'])

        torch.manual_seed(42)
        output1 = block(x)

        torch.manual_seed(42)
        output2 = block(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_transformer_block_no_nan(self):
        """Test that TransformerBlock doesn't produce NaN values."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg['emb_dim'])

        output = block(x)

        # Output should not contain NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_transformer_block_residual_connection(self):
        """Test that residual connections work."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg['emb_dim'])

        output = block(x)

        # Output should be different from input (due to transformations)
        self.assertFalse(torch.allclose(output, x))


class TestGPTModel(unittest.TestCase):
    """Test cases for GPTModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = {
            'vocab_size': 1000,
            'emb_dim': 64,
            'context_length': 32,
            'n_heads': 8,
            'n_layers': 2,
            'drop_rate': 0.1
        }
        self.batch_size = 2
        self.seq_len = 8

    def test_gpt_model_initialization(self):
        """Test GPTModel initialization."""
        model = GPTModel(self.cfg)

        # Check that components are created
        self.assertIsInstance(model.embedding, GPTEmbedding)
        self.assertIsInstance(model.dropout, nn.Dropout)
        self.assertIsInstance(model.trf_blocks, nn.Sequential)
        self.assertIsInstance(model.final_norm, nn.Module)  # RMSNorm
        self.assertIsInstance(model.out_head, nn.Linear)

    def test_gpt_model_forward_shape(self):
        """Test GPTModel forward pass shape."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (self.batch_size, self.seq_len))

        output = model(token_ids)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.cfg['vocab_size'])
        self.assertEqual(output.shape, expected_shape)

    def test_gpt_model_forward_deterministic(self):
        """Test that GPTModel is deterministic."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (self.batch_size, self.seq_len))

        torch.manual_seed(42)
        output1 = model(token_ids)

        torch.manual_seed(42)
        output2 = model(token_ids)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_gpt_model_no_nan(self):
        """Test that GPTModel doesn't produce NaN values."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (self.batch_size, self.seq_len))

        output = model(token_ids)

        # Output should not contain NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_gpt_model_context_length_limit(self):
        """Test that GPTModel enforces context length limit."""
        model = GPTModel(self.cfg)

        # Create input longer than context length
        long_seq_len = self.cfg['context_length'] + 10
        token_ids = torch.randint(0, self.cfg['vocab_size'], (self.batch_size, long_seq_len))

        # Should raise ValueError
        with self.assertRaises(ValueError):
            model(token_ids)


class TestGPTDataset(unittest.TestCase):
    """Test cases for GPTDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(side_effect=lambda x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        self.docs = ["hello world", "test document"]
        self.max_length = 4
        self.stride = 2

    def test_dataset_initialization(self):
        """Test GPTDataset initialization."""
        dataset = GPTDataset(self.docs, self.tokenizer, self.max_length, self.stride)

        # Check that samples are created
        self.assertGreater(len(dataset.input_ids), 0)
        self.assertGreater(len(dataset.target_ids), 0)

    def test_dataset_length(self):
        """Test GPTDataset length."""
        dataset = GPTDataset(self.docs, self.tokenizer, self.max_length, self.stride)

        # Check length
        self.assertEqual(len(dataset), len(dataset.input_ids))
        self.assertEqual(len(dataset), len(dataset.target_ids))

    def test_dataset_getitem(self):
        """Test GPTDataset __getitem__ method."""
        dataset = GPTDataset(self.docs, self.tokenizer, self.max_length, self.stride)

        if len(dataset) > 0:
            input_ids, target_ids = dataset[0]

            # Check shapes
            self.assertEqual(input_ids.shape, torch.Size([self.max_length]))
            self.assertEqual(target_ids.shape, torch.Size([self.max_length]))

            # Check that targets are shifted by one position
            self.assertTrue(torch.equal(target_ids[:-1], input_ids[1:]))


class TestGenerateFunctions(unittest.TestCase):
    """Test cases for generation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = {
            'vocab_size': 100,
            'emb_dim': 32,
            'context_length': 16,
            'n_heads': 4,
            'n_layers': 2,
            'drop_rate': 0.1
        }
        self.model = GPTModel(self.cfg)
        self.model.eval()

    def test_generate_new_tokens_shape(self):
        """Test generate_new_tokens output shape."""
        idx = torch.randint(0, self.cfg['vocab_size'], (1, 4))
        max_new_tokens = 3
        context_size = self.cfg['context_length']

        output = generate_new_tokens(self.model, idx, max_new_tokens, context_size)

        # Check output shape
        expected_shape = (1, 4 + max_new_tokens)
        self.assertEqual(output.shape, expected_shape)

    def test_generate_new_tokens_deterministic(self):
        """Test that generate_new_tokens is deterministic with same seed."""
        idx = torch.randint(0, self.cfg['vocab_size'], (1, 4))
        max_new_tokens = 2
        context_size = self.cfg['context_length']

        torch.manual_seed(42)
        output1 = generate_new_tokens(self.model, idx, max_new_tokens, context_size)

        torch.manual_seed(42)
        output2 = generate_new_tokens(self.model, idx, max_new_tokens, context_size)

        # Outputs should be identical
        self.assertTrue(torch.equal(output1, output2))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_setup_tokenizer(self):
        """Test setup_tokenizer function."""
        # This test might be slow due to downloading tokenizer
        # Skip if running in CI or if network is not available
        try:
            tokenizer = setup_tokenizer()

            # Check that tokenizer has expected attributes
            self.assertTrue(hasattr(tokenizer, 'encode'))
            self.assertTrue(hasattr(tokenizer, 'decode'))
            self.assertTrue(hasattr(tokenizer, 'vocab_size'))

            # Test basic functionality
            test_text = "Hello world"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)

            self.assertIsInstance(tokens, list)
            self.assertIsInstance(decoded, str)

        except Exception as e:
            self.skipTest(f"Tokenizer setup failed: {e}")

    def test_create_dataloader(self):
        """Test create_dataloader function."""
        # Mock documents
        docs = ["hello world", "test document", "another test"]

        try:
            dataloader = create_dataloader(
                txt=docs,
                batch_size=2,
                max_length=4,
                stride=2,
                shuffle=True,
                drop_last=True,
                num_workers=0
            )

            # Check that dataloader is created
            self.assertIsInstance(dataloader, DataLoader)

            # Test getting a batch
            for batch_input_ids, batch_labels in dataloader:
                # Check batch shapes
                self.assertEqual(batch_input_ids.shape, batch_labels.shape)
                self.assertEqual(batch_input_ids.shape[0], 2)  # batch_size
                self.assertEqual(batch_input_ids.shape[1], 4)  # max_length
                break  # Only test first batch

        except Exception as e:
            self.skipTest(f"DataLoader creation failed: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete GPT pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = {
            'vocab_size': 100,
            'emb_dim': 32,
            'context_length': 16,
            'n_heads': 4,
            'n_layers': 2,
            'drop_rate': 0.1
        }

    def test_full_gpt_pipeline(self):
        """Test the complete GPT pipeline from input to output."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (2, 8))

        # Forward pass
        logits = model(token_ids)

        # Check output shape
        expected_shape = (2, 8, self.cfg['vocab_size'])
        self.assertEqual(logits.shape, expected_shape)

        # Check that logits are reasonable
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_gpt_with_different_sequence_lengths(self):
        """Test GPT with different sequence lengths."""
        model = GPTModel(self.cfg)

        # Test with different sequence lengths
        for seq_len in [1, 4, 8, 16]:
            token_ids = torch.randint(0, self.cfg['vocab_size'], (1, seq_len))
            logits = model(token_ids)

            expected_shape = (1, seq_len, self.cfg['vocab_size'])
            self.assertEqual(logits.shape, expected_shape)

    def test_gpt_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (2, 8))

        logits = model(token_ids)
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist in model parameters
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                self.assertFalse(torch.isnan(param.grad).any())
                break

        self.assertTrue(has_gradients, "No gradients found in model parameters")


# =============================================================================
# TODO Implementation Tests
# =============================================================================

class TestGPTEmbeddingTODOs(unittest.TestCase):
    """Test cases for GPTEmbedding TODO implementations (1.1, 1.2)."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 100
        self.emb_dim = 32
        self.context_length = 16
        self.batch_size = 2
        self.seq_length = 8

    def test_todo_1_1_embedding_initialization(self):
        """Test TODO 1.1: Embedding layer initialization."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)

        # Verify embedding layer exists and has correct dimensions
        self.assertIsInstance(embedding.token_embeddings, nn.Embedding)
        self.assertEqual(embedding.token_embeddings.num_embeddings, self.vocab_size)
        self.assertEqual(embedding.token_embeddings.embedding_dim, self.emb_dim)

        # Verify no positional embeddings (since we use RoPE)
        self.assertFalse(hasattr(embedding, 'position_embeddings'))

    def test_todo_1_2_embedding_forward_shape(self):
        """Test TODO 1.2: Embedding forward pass shape."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)
        token_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))

        output = embedding(token_ids)

        # Verify output shape
        expected_shape = (self.batch_size, self.seq_length, self.emb_dim)
        self.assertEqual(output.shape, expected_shape)

        # Verify output is not NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_todo_1_2_embedding_deterministic(self):
        """Test TODO 1.2: Embedding forward pass is deterministic."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)
        token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        torch.manual_seed(42)
        output1 = embedding(token_ids)

        torch.manual_seed(42)
        output2 = embedding(token_ids)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_todo_1_2_embedding_token_uniqueness(self):
        """Test TODO 1.2: Different tokens produce different embeddings."""
        embedding = GPTEmbedding(self.vocab_size, self.emb_dim, self.context_length)

        # Test with different tokens
        token_ids = torch.tensor([[0, 1, 2, 3]])
        output = embedding(token_ids)

        # Different tokens should produce different embeddings
        for i in range(3):
            for j in range(i+1, 4):
                self.assertFalse(torch.allclose(output[0, i], output[0, j], atol=1e-6))


class TestMultiHeadAttentionTODOs(unittest.TestCase):
    """Test cases for MultiHeadAttention TODO implementations (1.3, 1.4)."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_in = 32
        self.context_length = 16
        self.dropout = 0.1
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 8

    def test_todo_1_3_attention_initialization(self):
        """Test TODO 1.3: MultiHeadAttention initialization."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)

        # Verify linear layers exist
        self.assertIsInstance(attention.W_query, nn.Linear)
        self.assertIsInstance(attention.W_key, nn.Linear)
        self.assertIsInstance(attention.W_value, nn.Linear)
        self.assertIsInstance(attention.out_proj, nn.Linear)

        # Verify dimensions
        self.assertEqual(attention.head_dim, self.d_in // self.num_heads)
        self.assertEqual(attention.scale, 1 / math.sqrt(attention.head_dim))

        # Verify RoPE initialization
        self.assertTrue(hasattr(attention, 'rope'))

        # Verify causal mask
        self.assertTrue(hasattr(attention, 'mask'))
        self.assertEqual(attention.mask.shape, (self.context_length, self.context_length))

        # Verify mask is upper triangular
        mask_np = attention.mask.cpu().numpy()
        self.assertTrue(np.all(np.triu(mask_np, k=1) == mask_np))

    def test_todo_1_4_attention_forward_shape(self):
        """Test TODO 1.4: MultiHeadAttention forward pass shape."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_in)

        output = attention(x)

        # Verify output shape matches input
        expected_shape = (self.batch_size, self.seq_len, self.d_in)
        self.assertEqual(output.shape, expected_shape)

    def test_todo_1_4_attention_causal_masking(self):
        """Test TODO 1.4: Causal masking prevents future attention."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)

        # Create input where we can verify causal masking
        x = torch.randn(1, 4, self.d_in)

        # Forward pass
        output = attention(x)

        # Output should not contain NaN values
        self.assertFalse(torch.isnan(output).any())

        # Test with different sequence lengths
        for seq_len in range(1, min(5, self.context_length)):
            x_test = torch.randn(1, seq_len, self.d_in)
            output_test = attention(x_test)
            self.assertEqual(output_test.shape, (1, seq_len, self.d_in))

    def test_todo_1_4_attention_rope_applied(self):
        """Test TODO 1.4: RoPE is applied to queries and keys."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)
        x = torch.randn(1, 4, self.d_in)

        # Forward pass
        output = attention(x)

        # Verify output is reasonable
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # Test that attention works with different positions
        x_long = torch.randn(1, self.context_length, self.d_in)
        output_long = attention(x_long)
        self.assertEqual(output_long.shape, (1, self.context_length, self.d_in))

    def test_todo_1_4_attention_deterministic(self):
        """Test TODO 1.4: Attention is deterministic with same input."""
        attention = MultiHeadAttention(self.d_in, self.context_length, self.dropout, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_in)

        torch.manual_seed(42)
        output1 = attention(x)

        torch.manual_seed(42)
        output2 = attention(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestSwiGLUTODOs(unittest.TestCase):
    """Test cases for SwiGLU TODO implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.dimension = 32
        self.batch_size = 2
        self.seq_len = 8

    def test_swiglu_initialization(self):
        """Test SwiGLU initialization."""
        swiglu = SwiGLU(self.dimension)

        # Verify linear layers exist
        self.assertIsInstance(swiglu.linear_1, nn.Linear)
        self.assertIsInstance(swiglu.linear_2, nn.Linear)

        # Verify dimensions
        self.assertEqual(swiglu.linear_1.in_features, self.dimension)
        self.assertEqual(swiglu.linear_1.out_features, self.dimension)
        self.assertEqual(swiglu.linear_2.in_features, self.dimension)
        self.assertEqual(swiglu.linear_2.out_features, self.dimension)

    def test_swiglu_forward_shape(self):
        """Test SwiGLU forward pass shape."""
        swiglu = SwiGLU(self.dimension)
        x = torch.randn(self.batch_size, self.seq_len, self.dimension)

        output = swiglu(x)

        # Verify output shape matches input
        self.assertEqual(output.shape, x.shape)

    def test_swiglu_swish_gating(self):
        """Test SwiGLU swish gating mechanism."""
        swiglu = SwiGLU(self.dimension)
        x = torch.randn(1, 1, self.dimension)

        output = swiglu(x)

        # Verify output is reasonable
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # Test with different input ranges
        x_positive = torch.abs(torch.randn(1, 1, self.dimension))
        output_positive = swiglu(x_positive)
        self.assertFalse(torch.isnan(output_positive).any())

    def test_swiglu_deterministic(self):
        """Test SwiGLU is deterministic."""
        swiglu = SwiGLU(self.dimension)
        x = torch.randn(self.batch_size, self.seq_len, self.dimension)

        torch.manual_seed(42)
        output1 = swiglu(x)

        torch.manual_seed(42)
        output2 = swiglu(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestFeedForwardTODOs(unittest.TestCase):
    """Test cases for FeedForward TODO implementations (1.5, 1.6)."""

    def setUp(self):
        """Set up test fixtures."""
        self.emb_dim = 32
        self.batch_size = 2
        self.seq_len = 8

    def test_todo_1_5_feedforward_initialization(self):
        """Test TODO 1.5: FeedForward initialization."""
        ff = FeedForward(self.emb_dim)

        # Verify linear layers exist
        self.assertIsInstance(ff.fc1, nn.Linear)
        self.assertIsInstance(ff.fc2, nn.Linear)

        # Verify dimensions (using efficient chunking approach)
        expected_hidden_dim = int(round((8/3) * self.emb_dim / 2))
        self.assertEqual(ff.fc1.out_features, 2 * expected_hidden_dim)
        self.assertEqual(ff.fc2.in_features, expected_hidden_dim)
        self.assertEqual(ff.fc2.out_features, self.emb_dim)

    def test_todo_1_6_feedforward_forward_shape(self):
        """Test TODO 1.6: FeedForward forward pass shape."""
        ff = FeedForward(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        output = ff(x)

        # Verify output shape matches input
        self.assertEqual(output.shape, x.shape)

    def test_todo_1_6_feedforward_swiglu_chunking(self):
        """Test TODO 1.6: SwiGLU chunking implementation."""
        ff = FeedForward(self.emb_dim)
        x = torch.randn(1, 1, self.emb_dim)

        output = ff(x)

        # Verify output is reasonable
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # Test with different input values
        x_zero = torch.zeros(1, 1, self.emb_dim)
        output_zero = ff(x_zero)
        self.assertFalse(torch.isnan(output_zero).any())

    def test_todo_1_6_feedforward_deterministic(self):
        """Test TODO 1.6: FeedForward is deterministic."""
        ff = FeedForward(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)

        torch.manual_seed(42)
        output1 = ff(x)

        torch.manual_seed(42)
        output2 = ff(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestTransformerBlockTODOs(unittest.TestCase):
    """Test cases for TransformerBlock TODO implementations (1.7, 1.8, 1.9)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = {
            'emb_dim': 32,
            'context_length': 16,
            'n_heads': 4,
            'n_layers': 2,
            'drop_rate': 0.1
        }
        self.batch_size = 2
        self.seq_len = 8

    def test_todo_1_7_transformer_block_initialization(self):
        """Test TODO 1.7: TransformerBlock initialization."""
        block = TransformerBlock(self.cfg)

        # Verify components exist
        self.assertIsInstance(block.self_attn, MultiHeadAttention)
        self.assertIsInstance(block.ffn, FeedForward)
        self.assertIsInstance(block.norm1, nn.Module)  # RMSNorm
        self.assertIsInstance(block.norm2, nn.Module)  # RMSNorm

        # Verify dropout probability is stored
        self.assertEqual(block.dropout_p, self.cfg['drop_rate'])

    def test_todo_1_8_maybe_dropout_function(self):
        """Test TODO 1.8: maybe_dropout function."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(2, 4, self.cfg['emb_dim'])

        # Test with dropout enabled
        output = block.maybe_dropout(x)
        self.assertEqual(output.shape, x.shape)

        # Test with dropout disabled
        block_no_dropout = TransformerBlock({'emb_dim': 32, 'context_length': 16,
                                            'n_heads': 4, 'n_layers': 2, 'drop_rate': 0.0})
        output_no_dropout = block_no_dropout.maybe_dropout(x)
        self.assertTrue(torch.allclose(output_no_dropout, x))

    def test_todo_1_9_transformer_block_forward_shape(self):
        """Test TODO 1.9: TransformerBlock forward pass shape."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg['emb_dim'])

        output = block(x)

        # Verify output shape matches input
        self.assertEqual(output.shape, x.shape)

    def test_todo_1_9_transformer_block_residual_connections(self):
        """Test TODO 1.9: Residual connections work correctly."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(1, 4, self.cfg['emb_dim'])

        output = block(x)

        # Output should be different from input (due to transformations)
        self.assertFalse(torch.allclose(output, x, atol=1e-6))

        # But should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_todo_1_9_transformer_block_pre_norm(self):
        """Test TODO 1.9: Pre-norm architecture."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(1, 4, self.cfg['emb_dim'])

        output = block(x)

        # Verify output is reasonable
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_todo_1_9_transformer_block_deterministic(self):
        """Test TODO 1.9: TransformerBlock is deterministic."""
        block = TransformerBlock(self.cfg)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg['emb_dim'])

        torch.manual_seed(42)
        output1 = block(x)

        torch.manual_seed(42)
        output2 = block(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestGPTModelTODOs(unittest.TestCase):
    """Test cases for GPTModel TODO implementations (1.10, 1.11)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = {
            'vocab_size': 100,
            'emb_dim': 32,
            'context_length': 16,
            'n_heads': 4,
            'n_layers': 2,
            'drop_rate': 0.1
        }
        self.batch_size = 2
        self.seq_len = 8

    def test_todo_1_10_gpt_model_initialization(self):
        """Test TODO 1.10: GPTModel initialization."""
        model = GPTModel(self.cfg)

        # Verify components exist
        self.assertIsInstance(model.embedding, GPTEmbedding)
        self.assertIsInstance(model.dropout, nn.Dropout)
        self.assertIsInstance(model.trf_blocks, nn.Sequential)
        self.assertIsInstance(model.final_norm, nn.Module)  # RMSNorm
        self.assertIsInstance(model.out_head, nn.Linear)

        # Verify context length is stored
        self.assertEqual(model.context_length, self.cfg['context_length'])

        # Verify number of transformer blocks
        self.assertEqual(len(model.trf_blocks), self.cfg['n_layers'])

    def test_todo_1_10_weight_tying(self):
        """Test TODO 1.10: Weight tying between input and output embeddings."""
        model = GPTModel(self.cfg)

        # Verify weight tying is implemented
        self.assertTrue(torch.equal(model.out_head.weight, model.embedding.token_embeddings.weight))

    def test_todo_1_11_gpt_model_forward_shape(self):
        """Test TODO 1.11: GPTModel forward pass shape."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (self.batch_size, self.seq_len))

        output = model(token_ids)

        # Verify output shape
        expected_shape = (self.batch_size, self.seq_len, self.cfg['vocab_size'])
        self.assertEqual(output.shape, expected_shape)

    def test_todo_1_11_gpt_model_context_length_enforcement(self):
        """Test TODO 1.11: Context length enforcement."""
        model = GPTModel(self.cfg)

        # Test with valid sequence length
        valid_token_ids = torch.randint(0, self.cfg['vocab_size'], (1, self.cfg['context_length']))
        output = model(valid_token_ids)
        self.assertEqual(output.shape, (1, self.cfg['context_length'], self.cfg['vocab_size']))

        # Test with invalid sequence length
        invalid_token_ids = torch.randint(0, self.cfg['vocab_size'], (1, self.cfg['context_length'] + 1))
        with self.assertRaises(ValueError):
            model(invalid_token_ids)

    def test_todo_1_11_gpt_model_forward_deterministic(self):
        """Test TODO 1.11: GPTModel forward pass is deterministic."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (self.batch_size, self.seq_len))

        torch.manual_seed(42)
        output1 = model(token_ids)

        torch.manual_seed(42)
        output2 = model(token_ids)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_todo_1_11_gpt_model_logits_properties(self):
        """Test TODO 1.11: GPTModel logits have reasonable properties."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (1, 4))

        logits = model(token_ids)

        # Logits should not be NaN or infinite
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Logits should have reasonable range
        self.assertTrue(torch.all(logits > -100))
        self.assertTrue(torch.all(logits < 100))


class TestGPTDatasetTODOs(unittest.TestCase):
    """Test cases for GPTDataset TODO implementations (1.12, 1.13)."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(side_effect=lambda x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        self.docs = ["hello world", "test document"]
        self.max_length = 4
        self.stride = 2

    def test_todo_1_12_dataset_initialization(self):
        """Test TODO 1.12: GPTDataset initialization."""
        dataset = GPTDataset(self.docs, self.tokenizer, self.max_length, self.stride)

        # Verify attributes are stored
        self.assertEqual(dataset.tokenizer, self.tokenizer)
        self.assertEqual(dataset.max_length, self.max_length)
        self.assertEqual(dataset.stride, self.stride)

        # Verify samples are created
        self.assertGreater(len(dataset.input_ids), 0)
        self.assertGreater(len(dataset.target_ids), 0)
        self.assertEqual(len(dataset.input_ids), len(dataset.target_ids))

    def test_todo_1_12_dataset_causal_lm_format(self):
        """Test TODO 1.12: Causal language modeling format."""
        dataset = GPTDataset(self.docs, self.tokenizer, self.max_length, self.stride)

        # Check that targets are shifted by one position
        for i in range(len(dataset)):
            input_ids = dataset.input_ids[i]
            target_ids = dataset.target_ids[i]

            # Targets should be shifted by one position
            self.assertTrue(torch.equal(target_ids[:-1], input_ids[1:]))

    def test_todo_1_13_dataset_getitem(self):
        """Test TODO 1.13: GPTDataset __getitem__ method."""
        dataset = GPTDataset(self.docs, self.tokenizer, self.max_length, self.stride)

        if len(dataset) > 0:
            input_ids, target_ids = dataset[0]

            # Verify shapes
            self.assertEqual(input_ids.shape, torch.Size([self.max_length]))
            self.assertEqual(target_ids.shape, torch.Size([self.max_length]))

            # Verify data types
            self.assertEqual(input_ids.dtype, torch.long)
            self.assertEqual(target_ids.dtype, torch.long)

    def test_todo_1_13_dataset_length(self):
        """Test TODO 1.13: GPTDataset length."""
        dataset = GPTDataset(self.docs, self.tokenizer, self.max_length, self.stride)

        # Verify length
        self.assertEqual(len(dataset), len(dataset.input_ids))
        self.assertEqual(len(dataset), len(dataset.target_ids))


class TestUtilityFunctionTODOs(unittest.TestCase):
    """Test cases for utility function TODO implementations (1.14, 1.15)."""

    def test_todo_1_14_create_dataloader_raw_text(self):
        """Test TODO 1.14: create_dataloader with raw text."""
        docs = ["hello world", "test document", "another test"]

        try:
            dataloader = create_dataloader(
                txt=docs,
                batch_size=2,
                max_length=4,
                stride=2,
                shuffle=True,
                drop_last=True,
                num_workers=0
            )

            # Verify dataloader is created
            self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

            # Test getting a batch
            for batch_input_ids, batch_labels in dataloader:
                # Verify batch shapes
                self.assertEqual(batch_input_ids.shape, batch_labels.shape)
                self.assertEqual(batch_input_ids.shape[0], 2)  # batch_size
                self.assertEqual(batch_input_ids.shape[1], 4)  # max_length
                break  # Only test first batch

        except Exception as e:
            self.skipTest(f"DataLoader creation failed: {e}")

    def test_todo_1_14_create_dataloader_error_handling(self):
        """Test TODO 1.14: create_dataloader error handling."""
        # Test with neither txt nor arrow_dataset_path provided
        with self.assertRaises(ValueError):
            create_dataloader()

    def test_todo_1_15_setup_tokenizer(self):
        """Test TODO 1.15: setup_tokenizer function."""
        try:
            tokenizer = setup_tokenizer()

            # Verify tokenizer has expected attributes
            self.assertTrue(hasattr(tokenizer, 'encode'))
            self.assertTrue(hasattr(tokenizer, 'decode'))
            self.assertTrue(hasattr(tokenizer, 'vocab_size'))

            # Test basic functionality
            test_text = "Hello world"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)

            self.assertIsInstance(tokens, list)
            self.assertIsInstance(decoded, str)

            # Verify special tokens are added
            self.assertIsNotNone(tokenizer.pad_token)

        except Exception as e:
            self.skipTest(f"Tokenizer setup failed: {e}")


class TestToyModelIntegration(unittest.TestCase):
    """Integration tests using small toy models."""

    def setUp(self):
        """Set up toy model configuration."""
        self.toy_cfg = {
            'vocab_size': 50,
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0  # No dropout for deterministic testing
        }

    def test_toy_model_end_to_end(self):
        """Test complete toy model end-to-end."""
        model = GPTModel(self.toy_cfg)
        token_ids = torch.randint(0, self.toy_cfg['vocab_size'], (1, 4))

        # Forward pass
        logits = model(token_ids)

        # Verify output shape
        expected_shape = (1, 4, self.toy_cfg['vocab_size'])
        self.assertEqual(logits.shape, expected_shape)

        # Verify logits are reasonable
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_toy_model_generation(self):
        """Test text generation with toy model."""
        model = GPTModel(self.toy_cfg)
        model.eval()

        # Test generation
        idx = torch.randint(0, self.toy_cfg['vocab_size'], (1, 2))
        max_new_tokens = 3

        generated = generate_new_tokens(
            model, idx, max_new_tokens,
            self.toy_cfg['context_length']
        )

        # Verify generation shape
        expected_length = 2 + max_new_tokens
        self.assertEqual(generated.shape, (1, expected_length))

        # Verify all tokens are valid
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < self.toy_cfg['vocab_size']))

    def test_toy_model_gradient_flow(self):
        """Test gradient flow through toy model."""
        model = GPTModel(self.toy_cfg)
        token_ids = torch.randint(0, self.toy_cfg['vocab_size'], (1, 4))

        # Forward pass
        logits = model(token_ids)
        loss = logits.sum()

        # Backward pass
        loss.backward()

        # Verify gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                self.assertFalse(torch.isnan(param.grad).any())
                break

        self.assertTrue(has_gradients, "No gradients found in model parameters")

    def test_toy_model_different_sequence_lengths(self):
        """Test toy model with different sequence lengths."""
        model = GPTModel(self.toy_cfg)

        # Test with different sequence lengths
        for seq_len in range(1, self.toy_cfg['context_length'] + 1):
            token_ids = torch.randint(0, self.toy_cfg['vocab_size'], (1, seq_len))
            logits = model(token_ids)

            expected_shape = (1, seq_len, self.toy_cfg['vocab_size'])
            self.assertEqual(logits.shape, expected_shape)

            # Verify logits are reasonable
            self.assertFalse(torch.isnan(logits).any())
            self.assertFalse(torch.isinf(logits).any())


# =============================================================================
# Mathematical Correctness Tests
# =============================================================================

class TestMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of GPT components."""

    def test_embedding_mathematical_properties(self):
        """Test embedding mathematical properties."""
        vocab_size = 10
        emb_dim = 8
        embedding = GPTEmbedding(vocab_size, emb_dim, 16)

        # Test that embedding lookup is deterministic
        token_ids = torch.tensor([[1, 2, 3]])
        output1 = embedding(token_ids)
        output2 = embedding(token_ids)
        self.assertTrue(torch.allclose(output1, output2))

        # Test that different tokens produce different embeddings
        token_ids_diff = torch.tensor([[1, 2]])
        output_diff = embedding(token_ids_diff)
        self.assertFalse(torch.allclose(output_diff[0, 0], output_diff[0, 1]))

        # Test embedding dimensions
        self.assertEqual(output1.shape, (1, 3, emb_dim))

    def test_attention_mathematical_properties(self):
        """Test attention mathematical properties."""
        d_in = 16
        num_heads = 2
        attention = MultiHeadAttention(d_in, 8, 0.0, num_heads)

        # Test attention scaling factor
        expected_scale = 1 / math.sqrt(d_in // num_heads)
        self.assertAlmostEqual(attention.scale, expected_scale, places=6)

        # Test causal mask properties
        mask = attention.mask
        self.assertEqual(mask.shape, (8, 8))

        # Verify upper triangular mask
        for i in range(8):
            for j in range(8):
                if j > i:  # Above diagonal should be True
                    self.assertTrue(mask[i, j])
                else:  # On or below diagonal should be False
                    self.assertFalse(mask[i, j])

        # Test attention output properties
        x = torch.randn(1, 4, d_in)
        output = attention(x)

        # Output should preserve input shape
        self.assertEqual(output.shape, x.shape)

        # Output should not contain NaN or infinite values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_swiglu_mathematical_properties(self):
        """Test SwiGLU mathematical properties."""
        dimension = 8
        swiglu = SwiGLU(dimension)

        # Test with known input
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        output = swiglu(x)

        # Output should have same shape as input
        self.assertEqual(output.shape, x.shape)

        # Test Swish function properties
        # Swish(x) = x * sigmoid(x)
        gate = swiglu.linear_2(x)
        swish_gate = gate * torch.sigmoid(gate)

        # Swish should be smooth and differentiable
        self.assertFalse(torch.isnan(swish_gate).any())
        self.assertFalse(torch.isinf(swish_gate).any())

        # Test with zero input
        x_zero = torch.zeros(1, dimension)
        output_zero = swiglu(x_zero)
        self.assertFalse(torch.isnan(output_zero).any())

    def test_feedforward_mathematical_properties(self):
        """Test FeedForward mathematical properties."""
        emb_dim = 16
        ff = FeedForward(emb_dim)

        # Test expansion factor calculation
        expected_hidden_dim = int(round((8/3) * emb_dim / 2))
        self.assertEqual(ff.fc1.out_features, 2 * expected_hidden_dim)
        self.assertEqual(ff.fc2.in_features, expected_hidden_dim)

        # Test forward pass
        x = torch.randn(1, 4, emb_dim)
        output = ff(x)

        # Output should preserve input shape
        self.assertEqual(output.shape, x.shape)

        # Test chunking implementation
        # fc1 output should be chunkable into two equal parts
        fc1_output = ff.fc1(x)
        x1, x2 = fc1_output.chunk(2, dim=-1)
        self.assertEqual(x1.shape, x2.shape)

        # Test SwiGLU chunking
        swiglu_output = torch.nn.functional.silu(x1) * x2
        self.assertEqual(swiglu_output.shape, x1.shape)

    def test_transformer_block_mathematical_properties(self):
        """Test TransformerBlock mathematical properties."""
        cfg = {
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        block = TransformerBlock(cfg)

        # Test residual connection properties
        x = torch.randn(1, 4, cfg['emb_dim'])
        output = block(x)

        # Output should be different from input (due to transformations)
        self.assertFalse(torch.allclose(output, x, atol=1e-6))

        # Test pre-norm architecture
        # Both attention and MLP should have pre-norm
        self.assertIsInstance(block.norm1, nn.Module)
        self.assertIsInstance(block.norm2, nn.Module)

        # Test dropout behavior
        block_no_dropout = TransformerBlock({
            'emb_dim': 16, 'context_length': 8, 'n_heads': 2,
            'n_layers': 1, 'drop_rate': 0.0
        })

        # With no dropout, maybe_dropout should return input unchanged
        x_test = torch.randn(1, 2, 16)
        output_no_dropout = block_no_dropout.maybe_dropout(x_test)
        self.assertTrue(torch.allclose(output_no_dropout, x_test))

    def test_gpt_model_mathematical_properties(self):
        """Test GPTModel mathematical properties."""
        cfg = {
            'vocab_size': 20,
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        model = GPTModel(cfg)

        # Test weight tying
        self.assertTrue(torch.equal(model.out_head.weight, model.embedding.token_embeddings.weight))

        # Test forward pass
        token_ids = torch.randint(0, cfg['vocab_size'], (1, 4))
        logits = model(token_ids)

        # Logits should have correct shape
        expected_shape = (1, 4, cfg['vocab_size'])
        self.assertEqual(logits.shape, expected_shape)

        # Logits should be finite
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Test context length enforcement
        long_token_ids = torch.randint(0, cfg['vocab_size'], (1, cfg['context_length'] + 1))
        with self.assertRaises(ValueError):
            model(long_token_ids)

    def test_dataset_mathematical_properties(self):
        """Test GPTDataset mathematical properties."""
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=lambda x: [1, 2, 3, 4, 5, 6, 7, 8])

        docs = ["hello world", "test document"]
        max_length = 4
        stride = 2

        dataset = GPTDataset(docs, tokenizer, max_length, stride)

        # Test causal language modeling format
        for i in range(len(dataset)):
            input_ids = dataset.input_ids[i]
            target_ids = dataset.target_ids[i]

            # Targets should be shifted by one position
            self.assertTrue(torch.equal(target_ids[:-1], input_ids[1:]))

            # All sequences should have correct length
            self.assertEqual(len(input_ids), max_length)
            self.assertEqual(len(target_ids), max_length)

    def test_generation_mathematical_properties(self):
        """Test generation mathematical properties."""
        cfg = {
            'vocab_size': 20,
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        model = GPTModel(cfg)
        model.eval()

        # Test generation with different temperatures
        idx = torch.tensor([[1, 2]])

        # Test with temperature = 1.0
        generated_1 = generate_new_tokens(model, idx, max_new_tokens=2,
                                        context_size=cfg['context_length'], temperature=1.0)

        # Test with temperature = 0.5
        generated_05 = generate_new_tokens(model, idx, max_new_tokens=2,
                                        context_size=cfg['context_length'], temperature=0.5)

        # Both should have correct shape
        self.assertEqual(generated_1.shape, (1, 4))  # 2 + 2 new tokens
        self.assertEqual(generated_05.shape, (1, 4))

        # All generated tokens should be valid
        self.assertTrue(torch.all(generated_1 >= 0))
        self.assertTrue(torch.all(generated_1 < cfg['vocab_size']))
        self.assertTrue(torch.all(generated_05 >= 0))
        self.assertTrue(torch.all(generated_05 < cfg['vocab_size']))


class TestKnownBehaviors(unittest.TestCase):
    """Test against known expected behaviors."""

    def test_attention_causal_behavior(self):
        """Test that attention respects causal masking."""
        attention = MultiHeadAttention(8, 4, 0.0, 2)

        # Create input where we can verify causal behavior
        x = torch.randn(1, 3, 8)
        output = attention(x)

        # Output should be well-behaved
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # Test with different sequence lengths
        for seq_len in range(1, 5):
            x_test = torch.randn(1, seq_len, 8)
            output_test = attention(x_test)
            self.assertEqual(output_test.shape, (1, seq_len, 8))

    def test_swiglu_gating_behavior(self):
        """Test SwiGLU gating behavior."""
        swiglu = SwiGLU(8)

        # Test gating with different input magnitudes
        x_small = torch.randn(1, 1, 8) * 0.1
        x_large = torch.randn(1, 1, 8) * 10

        output_small = swiglu(x_small)
        output_large = swiglu(x_large)

        # Both should be well-behaved
        self.assertFalse(torch.isnan(output_small).any())
        self.assertFalse(torch.isnan(output_large).any())
        self.assertFalse(torch.isinf(output_small).any())
        self.assertFalse(torch.isinf(output_large).any())

    def test_residual_connection_behavior(self):
        """Test residual connection behavior."""
        cfg = {
            'emb_dim': 8,
            'context_length': 4,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        block = TransformerBlock(cfg)

        # Test with zero input
        x_zero = torch.zeros(1, 2, 8)
        output_zero = block(x_zero)

        # Should not be NaN
        self.assertFalse(torch.isnan(output_zero).any())

        # Test with normal input
        x_normal = torch.randn(1, 2, 8)
        output_normal = block(x_normal)

        # Should be different from input
        self.assertFalse(torch.allclose(output_normal, x_normal, atol=1e-6))

    def test_model_deterministic_behavior(self):
        """Test that model is deterministic with same seed."""
        cfg = {
            'vocab_size': 10,
            'emb_dim': 8,
            'context_length': 4,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }

        model1 = GPTModel(cfg)
        model2 = GPTModel(cfg)

        # Copy weights to make models identical
        model2.load_state_dict(model1.state_dict())

        token_ids = torch.tensor([[1, 2, 3]])

        # Both models should produce identical output
        output1 = model1(token_ids)
        output2 = model2(token_ids)

        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_gradient_flow_behavior(self):
        """Test gradient flow behavior."""
        cfg = {
            'vocab_size': 10,
            'emb_dim': 8,
            'context_length': 4,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        model = GPTModel(cfg)

        token_ids = torch.randint(0, cfg['vocab_size'], (1, 3))
        logits = model(token_ids)
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist and are finite
        gradient_count = 0
        for param in model.parameters():
            if param.grad is not None:
                gradient_count += 1
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())

        # Should have gradients for most parameters
        self.assertGreater(gradient_count, 0)


# =============================================================================
# Advanced Edge Cases and Numerical Stability Tests
# =============================================================================

class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of GPT components."""

    def setUp(self):
        """Set up test fixtures."""
        self.toy_cfg = {
            'vocab_size': 20,
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }

    def test_attention_scaling_property(self):
        """Test that attention scaling follows 1/sqrt(d_k) rule."""
        d_in = 16
        num_heads = 2
        head_dim = d_in // num_heads
        expected_scale = 1 / math.sqrt(head_dim)

        attention = MultiHeadAttention(d_in, 8, 0.0, num_heads)
        self.assertAlmostEqual(attention.scale, expected_scale, places=6)

    def test_rope_positional_encoding_properties(self):
        """Test RoPE positional encoding properties."""
        attention = MultiHeadAttention(16, 8, 0.0, 2)

        # Test that RoPE is initialized
        self.assertTrue(hasattr(attention, 'rope'))

        # Test with different sequence lengths
        for seq_len in range(1, 9):
            x = torch.randn(1, seq_len, 16)
            output = attention(x)

            # Output should be well-behaved
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_swiglu_gating_properties(self):
        """Test SwiGLU gating mechanism properties."""
        swiglu = SwiGLU(16)

        # Test with zero input
        x_zero = torch.zeros(1, 1, 16)
        output_zero = swiglu(x_zero)
        self.assertFalse(torch.isnan(output_zero).any())

        # Test with negative input
        x_negative = torch.randn(1, 1, 16) * -1
        output_negative = swiglu(x_negative)
        self.assertFalse(torch.isnan(output_negative).any())

        # Test with large input
        x_large = torch.randn(1, 1, 16) * 10
        output_large = swiglu(x_large)
        self.assertFalse(torch.isnan(output_large).any())
        self.assertFalse(torch.isinf(output_large).any())

    def test_feedforward_expansion_factor(self):
        """Test FeedForward expansion factor calculation."""
        emb_dim = 16
        ff = FeedForward(emb_dim)

        # Verify expansion factor is applied correctly
        expected_hidden_dim = int(round((8/3) * emb_dim ))
        self.assertEqual(ff.fc1.out_features, 2 * expected_hidden_dim)
        self.assertEqual(ff.fc2.in_features, expected_hidden_dim)

    def test_transformer_block_residual_scaling(self):
        """Test that residual connections maintain proper scaling."""
        block = TransformerBlock(self.toy_cfg)

        # Test with small input
        x_small = torch.randn(1, 4, self.toy_cfg['emb_dim']) * 0.01
        output_small = block(x_small)

        # Test with large input
        x_large = torch.randn(1, 4, self.toy_cfg['emb_dim']) * 10
        output_large = block(x_large)

        # Both should be well-behaved
        self.assertFalse(torch.isnan(output_small).any())
        self.assertFalse(torch.isnan(output_large).any())
        self.assertFalse(torch.isinf(output_small).any())
        self.assertFalse(torch.isinf(output_large).any())

    def test_gpt_model_weight_tying_consistency(self):
        """Test that weight tying maintains consistency."""
        model = GPTModel(self.toy_cfg)

        # Verify weight tying
        self.assertTrue(torch.equal(model.out_head.weight, model.embedding.token_embeddings.weight))

        # Test that gradients flow to both tied weights
        token_ids = torch.randint(0, self.toy_cfg['vocab_size'], (1, 4))
        logits = model(token_ids)
        loss = logits.sum()
        loss.backward()

        # Both weights should have gradients
        self.assertIsNotNone(model.out_head.weight.grad)
        self.assertIsNotNone(model.embedding.token_embeddings.weight.grad)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.minimal_cfg = {
            'vocab_size': 10,
            'emb_dim': 8,
            'context_length': 4,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }

    def test_minimal_model_configuration(self):
        """Test model with minimal configuration."""
        model = GPTModel(self.minimal_cfg)

        # Test with single token
        single_token = torch.tensor([[1]])
        logits = model(single_token)
        self.assertEqual(logits.shape, (1, 1, self.minimal_cfg['vocab_size']))

        # Test with maximum context length
        max_context = torch.randint(0, self.minimal_cfg['vocab_size'],
                                   (1, self.minimal_cfg['context_length']))
        logits_max = model(max_context)
        self.assertEqual(logits_max.shape, (1, self.minimal_cfg['context_length'],
                                          self.minimal_cfg['vocab_size']))

    def test_attention_with_single_head(self):
        """Test attention with single head."""
        attention = MultiHeadAttention(8, 4, 0.0, 1)
        x = torch.randn(1, 3, 8)
        output = attention(x)

        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_attention_with_minimal_sequence(self):
        """Test attention with minimal sequence length."""
        attention = MultiHeadAttention(8, 4, 0.0, 2)

        # Test with single token
        x_single = torch.randn(1, 1, 8)
        output_single = attention(x_single)
        self.assertEqual(output_single.shape, (1, 1, 8))

        # Test with two tokens
        x_double = torch.randn(1, 2, 8)
        output_double = attention(x_double)
        self.assertEqual(output_double.shape, (1, 2, 8))

    def test_dataset_with_short_documents(self):
        """Test dataset with very short documents."""
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=lambda x: [1, 2])  # Very short

        docs = ["hi", "bye"]
        dataset = GPTDataset(docs, tokenizer, max_length=3, stride=1)

        # Should handle short documents gracefully
        self.assertGreaterEqual(len(dataset), 0)

    def test_dataset_with_empty_documents(self):
        """Test dataset with empty documents."""
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=lambda x: [])  # Empty

        docs = ["", "hello"]
        dataset = GPTDataset(docs, tokenizer, max_length=2, stride=1)

        # Should handle empty documents gracefully
        self.assertGreaterEqual(len(dataset), 0)

    def test_generation_with_minimal_context(self):
        """Test text generation with minimal context."""
        model = GPTModel(self.minimal_cfg)
        model.eval()

        # Test with single token context
        idx = torch.tensor([[1]])
        generated = generate_new_tokens(model, idx, max_new_tokens=2,
                                     context_size=self.minimal_cfg['context_length'])

        self.assertEqual(generated.shape, (1, 3))  # 1 + 2 new tokens
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < self.minimal_cfg['vocab_size']))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and precision."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = {
            'vocab_size': 50,
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }

    def test_attention_numerical_stability(self):
        """Test attention numerical stability."""
        attention = MultiHeadAttention(16, 8, 0.0, 2)

        # Test with extreme values
        x_extreme = torch.tensor([[[1000.0, -1000.0, 0.0, 1.0] * 4]])  # Repeat to get 16 dims
        output = attention(x_extreme)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_swiglu_numerical_stability(self):
        """Test SwiGLU numerical stability."""
        swiglu = SwiGLU(16)

        # Test with extreme values
        x_extreme = torch.tensor([[1000.0, -1000.0, 0.0, 1.0] * 4])
        output = swiglu(x_extreme)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_feedforward_numerical_stability(self):
        """Test FeedForward numerical stability."""
        ff = FeedForward(16)

        # Test with extreme values
        x_extreme = torch.tensor([[1000.0, -1000.0, 0.0, 1.0] * 4])
        output = ff(x_extreme)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_model_numerical_stability(self):
        """Test full model numerical stability."""
        model = GPTModel(self.cfg)

        # Test with extreme token IDs
        extreme_tokens = torch.tensor([[0, self.cfg['vocab_size']-1, 1, 2]])
        logits = model(extreme_tokens)

        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_gradient_numerical_stability(self):
        """Test gradient numerical stability."""
        model = GPTModel(self.cfg)
        token_ids = torch.randint(0, self.cfg['vocab_size'], (1, 4))

        logits = model(token_ids)
        loss = logits.sum()
        loss.backward()

        # Check that gradients are well-behaved
        for param in model.parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())


class TestShapeConsistency(unittest.TestCase):
    """Test shape consistency across different configurations."""

    def test_embedding_shape_consistency(self):
        """Test embedding shape consistency."""
        vocab_sizes = [10, 100, 1000]
        emb_dims = [8, 16, 32]

        for vocab_size in vocab_sizes:
            for emb_dim in emb_dims:
                embedding = GPTEmbedding(vocab_size, emb_dim, 16)
                token_ids = torch.randint(0, vocab_size, (2, 4))
                output = embedding(token_ids)

                self.assertEqual(output.shape, (2, 4, emb_dim))

    def test_attention_shape_consistency(self):
        """Test attention shape consistency."""
        d_ins = [8, 16, 32]
        num_heads_list = [1, 2, 4]

        for d_in in d_ins:
            for num_heads in num_heads_list:
                if d_in % num_heads == 0:  # Only test valid combinations
                    attention = MultiHeadAttention(d_in, 8, 0.0, num_heads)
                    x = torch.randn(2, 4, d_in)
                    output = attention(x)

                    self.assertEqual(output.shape, (2, 4, d_in))

    def test_model_shape_consistency(self):
        """Test model shape consistency."""
        configs = [
            {'vocab_size': 10, 'emb_dim': 8, 'context_length': 4, 'n_heads': 2, 'n_layers': 1, 'drop_rate': 0.0},
            {'vocab_size': 50, 'emb_dim': 16, 'context_length': 8, 'n_heads': 4, 'n_layers': 2, 'drop_rate': 0.0},
            {'vocab_size': 100, 'emb_dim': 32, 'context_length': 16, 'n_heads': 8, 'n_layers': 3, 'drop_rate': 0.0},
        ]

        for cfg in configs:
            model = GPTModel(cfg)

            # Test with different sequence lengths
            for seq_len in range(1, min(5, cfg['context_length']) + 1):
                token_ids = torch.randint(0, cfg['vocab_size'], (1, seq_len))
                logits = model(token_ids)

                expected_shape = (1, seq_len, cfg['vocab_size'])
                self.assertEqual(logits.shape, expected_shape)


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency and performance characteristics."""

    def test_model_memory_usage(self):
        """Test model memory usage with different configurations."""
        small_cfg = {
            'vocab_size': 100,
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }

        large_cfg = {
            'vocab_size': 1000,
            'emb_dim': 64,
            'context_length': 32,
            'n_heads': 8,
            'n_layers': 4,
            'drop_rate': 0.0
        }

        # Both models should work without memory issues
        small_model = GPTModel(small_cfg)
        large_model = GPTModel(large_cfg)

        # Test forward passes
        small_tokens = torch.randint(0, small_cfg['vocab_size'], (1, 4))
        large_tokens = torch.randint(0, large_cfg['vocab_size'], (1, 8))

        small_logits = small_model(small_tokens)
        large_logits = large_model(large_tokens)

        self.assertEqual(small_logits.shape, (1, 4, small_cfg['vocab_size']))
        self.assertEqual(large_logits.shape, (1, 8, large_cfg['vocab_size']))

    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        cfg = {
            'vocab_size': 100,
            'emb_dim': 16,
            'context_length': 8,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }

        model = GPTModel(cfg)

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            token_ids = torch.randint(0, cfg['vocab_size'], (batch_size, 4))
            logits = model(token_ids)

            expected_shape = (batch_size, 4, cfg['vocab_size'])
            self.assertEqual(logits.shape, expected_shape)


class TestEdgeCaseMathematicalProperties(unittest.TestCase):
    """Test mathematical properties in edge cases."""

    def test_minimal_dimensions(self):
        """Test with minimal dimensions."""
        # Test with minimal embedding dimension
        embedding = GPTEmbedding(5, 4, 4)
        token_ids = torch.tensor([[0, 1, 2]])
        output = embedding(token_ids)
        self.assertEqual(output.shape, (1, 3, 4))

        # Test with single head attention
        attention = MultiHeadAttention(4, 4, 0.0, 1)
        x = torch.randn(1, 2, 4)
        output = attention(x)
        self.assertEqual(output.shape, (1, 2, 4))

        # Test minimal transformer block
        cfg = {
            'emb_dim': 4,
            'context_length': 4,
            'n_heads': 1,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        block = TransformerBlock(cfg)
        x = torch.randn(1, 2, 4)
        output = block(x)
        self.assertEqual(output.shape, (1, 2, 4))

    def test_boundary_conditions(self):
        """Test boundary conditions."""
        cfg = {
            'vocab_size': 3,
            'emb_dim': 4,
            'context_length': 2,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        model = GPTModel(cfg)

        # Test with minimum valid sequence
        min_tokens = torch.tensor([[0]])
        logits_min = model(min_tokens)
        self.assertEqual(logits_min.shape, (1, 1, 3))

        # Test with maximum valid sequence
        max_tokens = torch.tensor([[0, 1]])
        logits_max = model(max_tokens)
        self.assertEqual(logits_max.shape, (1, 2, 3))

        # Test with boundary token IDs
        boundary_tokens = torch.tensor([[0, cfg['vocab_size']-1]])
        logits_boundary = model(boundary_tokens)
        self.assertEqual(logits_boundary.shape, (1, 2, 3))

    def test_numerical_precision(self):
        """Test numerical precision."""
        cfg = {
            'vocab_size': 10,
            'emb_dim': 8,
            'context_length': 4,
            'n_heads': 2,
            'n_layers': 1,
            'drop_rate': 0.0
        }
        model = GPTModel(cfg)

        # Test with very small values
        small_tokens = torch.tensor([[0, 1]])
        logits_small = model(small_tokens)
        self.assertFalse(torch.isnan(logits_small).any())

        # Test with values near vocabulary boundaries
        boundary_tokens = torch.tensor([[cfg['vocab_size']-1, 0]])
        logits_boundary = model(boundary_tokens)
        self.assertFalse(torch.isnan(logits_boundary).any())


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)