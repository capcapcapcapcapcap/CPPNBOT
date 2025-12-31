"""Numerical feature encoder for bot detection model."""

import torch
import torch.nn as nn


class NumericalEncoder(nn.Module):
    """数值特征编码器: 5维 → 64维
    
    Two-layer MLP with ReLU activation and LayerNorm.
    Architecture: input_dim → hidden_dim → output_dim
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 32,
        output_dim: int = 64
    ):
        """
        Args:
            input_dim: Input feature dimension (default: 5)
            hidden_dim: Hidden layer dimension (default: 32)
            output_dim: Output embedding dimension (default: 64)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Two-layer MLP: 5 → 32 → 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Layer normalization on output
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode numerical features.
        
        Args:
            x: Tensor[batch, input_dim] numerical features
            
        Returns:
            Tensor[batch, output_dim] encoded embeddings
        """
        # First layer with ReLU
        x = self.fc1(x)
        x = self.relu(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.relu(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x
