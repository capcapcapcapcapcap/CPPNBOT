"""Multi-modal feature fusion module for bot detection model."""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """多模态特征融合: (64 + 32) → 256
    
    Concatenates numerical and categorical embeddings,
    then projects to output dimension with dropout.
    """
    
    def __init__(
        self,
        num_dim: int = 64,
        cat_dim: int = 32,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            num_dim: Numerical embedding dimension (default: 64)
            cat_dim: Categorical embedding dimension (default: 32)
            output_dim: Output dimension (default: 256)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        
        self.num_dim = num_dim
        self.cat_dim = cat_dim
        self.output_dim = output_dim
        
        # Projection layer: concatenated → output_dim
        concat_dim = num_dim + cat_dim
        self.projection = nn.Linear(concat_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        num_embed: torch.Tensor,
        cat_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse numerical and categorical embeddings.
        
        Args:
            num_embed: Tensor[batch, num_dim] numerical embeddings
            cat_embed: Tensor[batch, cat_dim] categorical embeddings
            
        Returns:
            Tensor[batch, output_dim] fused representation
        """
        # Concatenate embeddings
        concat = torch.cat([num_embed, cat_embed], dim=1)
        
        # Project to output dimension
        output = self.projection(concat)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
