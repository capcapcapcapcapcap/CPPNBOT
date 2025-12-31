"""Categorical feature encoder for bot detection model."""

from typing import List

import torch
import torch.nn as nn


class CategoricalEncoder(nn.Module):
    """分类特征编码器: 3维 → 32维
    
    Uses separate embedding tables for each categorical feature,
    concatenates embeddings, and projects to output dimension.
    """
    
    def __init__(
        self,
        num_categories: List[int],
        embedding_dim: int = 16,
        output_dim: int = 32
    ):
        """
        Args:
            num_categories: List of category counts for each feature
            embedding_dim: Embedding dimension for each feature (default: 16)
            output_dim: Output embedding dimension (default: 32)
        """
        super().__init__()
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_features = len(num_categories)
        
        # Create separate embedding table for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, embedding_dim)
            for num_cat in num_categories
        ])
        
        # Projection layer: concatenated embeddings → output_dim
        concat_dim = self.num_features * embedding_dim
        self.projection = nn.Linear(concat_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode categorical features.
        
        Args:
            x: Tensor[batch, num_features] categorical features (integer indices)
            
        Returns:
            Tensor[batch, output_dim] encoded embeddings
        """
        # Get embeddings for each feature
        embedded = []
        for i, emb in enumerate(self.embeddings):
            embedded.append(emb(x[:, i]))
        
        # Concatenate all embeddings
        concat = torch.cat(embedded, dim=1)
        
        # Project to output dimension
        output = self.projection(concat)
        
        return output
