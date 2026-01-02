"""Multi-modal feature fusion module for bot detection model."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """注意力多模态融合: (64 + 32 + 256 + 128) → 256
    
    Uses attention-based weighting to balance modality contributions.
    Supports configurable modality combinations.
    
    Requirements:
        - 8.1: Output 256-dimensional fused representation
        - 8.2: Support configurable modality combinations
        - 8.3: Apply attention-based weighting
        - 8.4: Apply dropout for regularization
    """
    
    def __init__(
        self,
        num_dim: int = 64,
        cat_dim: int = 32,
        text_dim: int = 256,
        graph_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.1,
        enabled_modalities: Optional[List[str]] = None
    ):
        """
        Args:
            num_dim: Numerical embedding dimension (default: 64)
            cat_dim: Categorical embedding dimension (default: 32)
            text_dim: Text embedding dimension (default: 256)
            graph_dim: Graph embedding dimension (default: 128)
            output_dim: Output dimension (default: 256)
            dropout: Dropout probability (default: 0.1)
            enabled_modalities: List of enabled modalities (default: ['num', 'cat', 'text', 'graph'])
        """
        super().__init__()
        
        self.num_dim = num_dim
        self.cat_dim = cat_dim
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        
        # Default to all modalities if not specified
        if enabled_modalities is None:
            enabled_modalities = ['num', 'cat', 'text', 'graph']
        self.enabled_modalities = enabled_modalities
        
        # Modality dimension mapping
        self.modality_dims = {
            'num': num_dim,
            'cat': cat_dim,
            'text': text_dim,
            'graph': graph_dim
        }
        
        # Create projection layers for each enabled modality to common dimension
        self.modality_projections = nn.ModuleDict()
        for modality in self.enabled_modalities:
            dim = self.modality_dims[modality]
            self.modality_projections[modality] = nn.Linear(dim, output_dim)
        
        # Attention mechanism: learn attention weights for each modality
        # Each modality gets a learnable attention score
        self.attention_weights = nn.ParameterDict()
        for modality in self.enabled_modalities:
            self.attention_weights[modality] = nn.Parameter(torch.ones(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Store last computed attention weights for inspection
        self._last_attention_weights: Dict[str, float] = {}
    
    def forward(
        self,
        num_embed: Optional[torch.Tensor] = None,
        cat_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        graph_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse embeddings from multiple modalities using attention weighting.
        
        Args:
            num_embed: Tensor[batch, num_dim] numerical embeddings (optional)
            cat_embed: Tensor[batch, cat_dim] categorical embeddings (optional)
            text_embed: Tensor[batch, text_dim] text embeddings (optional)
            graph_embed: Tensor[batch, graph_dim] graph embeddings (optional)
            
        Returns:
            Tensor[batch, output_dim] fused representation
        """
        # Map modality names to embeddings
        embeddings = {
            'num': num_embed,
            'cat': cat_embed,
            'text': text_embed,
            'graph': graph_embed
        }
        
        # Collect available embeddings (enabled and provided)
        available_embeddings = {}
        for modality in self.enabled_modalities:
            embed = embeddings.get(modality)
            if embed is not None:
                available_embeddings[modality] = embed
        
        # Handle case where no embeddings are provided
        if not available_embeddings:
            raise ValueError("At least one modality embedding must be provided")
        
        # Get batch size from first available embedding
        first_embed = next(iter(available_embeddings.values()))
        batch_size = first_embed.size(0)
        device = first_embed.device
        
        # Project each modality to common dimension
        projected = {}
        for modality, embed in available_embeddings.items():
            projected[modality] = self.modality_projections[modality](embed)
        
        # Compute attention weights using softmax over available modalities
        raw_weights = []
        modality_order = list(projected.keys())
        for modality in modality_order:
            raw_weights.append(self.attention_weights[modality])
        
        # Stack and apply softmax to get normalized attention weights
        raw_weights_tensor = torch.cat(raw_weights)
        attention_probs = F.softmax(raw_weights_tensor, dim=0)
        
        # Store attention weights for inspection
        self._last_attention_weights = {}
        for i, modality in enumerate(modality_order):
            self._last_attention_weights[modality] = attention_probs[i].item()
        
        # Weighted sum of projected embeddings
        fused = torch.zeros(batch_size, self.output_dim, device=device)
        for i, modality in enumerate(modality_order):
            weight = attention_probs[i]
            fused = fused + weight * projected[modality]
        
        # Apply layer normalization
        fused = self.layer_norm(fused)
        
        # Apply dropout
        fused = self.dropout(fused)
        
        return fused
    
    def get_attention_weights(self) -> Dict[str, float]:
        """Return the last computed attention weights for each modality.
        
        Returns:
            Dictionary mapping modality names to their attention weights.
            Weights sum to 1.0 (within numerical tolerance).
        """
        return self._last_attention_weights.copy()


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
