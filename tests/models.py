"""Test models for FSDP testing."""

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    """Simple toy model for testing.
    
    Architecture:
    - Linear(10, 10) without bias
    - ReLU
    - Linear(10, 5) without bias
    """
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5, bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ToyModelWithBias(nn.Module):
    """Toy model with biases for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5, bias=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleTransformerBlock(nn.Module):
    """Simplified Transformer block for testing.
    
    This is a minimal version for testing FSDP with Transformer-like architectures.
    """
    
    def __init__(self, d_model=64, n_heads=4, d_ff=256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # FFN
        self.ffn1 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=False)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention (simplified - no masking)
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Scaled dot-product attention (simplified)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = self.out_proj(attn_out)
        
        # Add & Norm
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn2(torch.relu(self.ffn1(x)))
        
        # Add & Norm
        x = self.norm2(x + ffn_out)
        
        return x


class SimpleTransformer(nn.Module):
    """Simple Transformer model for testing."""
    
    def __init__(self, n_layers=4, d_model=64, n_heads=4, d_ff=256, vocab_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        # x: [batch_size, seq_len] (token indices)
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output(x)  # [batch_size, seq_len, vocab_size]
        return x

