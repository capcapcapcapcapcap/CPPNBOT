"""Multi-modal encoder combining numerical, categorical, and fusion modules."""

from typing import Dict, List, Union

import torch
import torch.nn as nn

from .encoders.numerical import NumericalEncoder
from .encoders.categorical import CategoricalEncoder
from .fusion import FusionModule


class MultiModalEncoder(nn.Module):
    """完整的多模态编码器
    
    Combines NumericalEncoder, CategoricalEncoder, and FusionModule
    to produce unified user embeddings.
    
    支持不同数据集的分类特征维度:
    - Twibot-20: 5维二值特征 (嵌入模式)
    - Misbot: 20维one-hot特征 (线性模式)
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with keys:
                - num_input_dim: Numerical input dimension (default: 8)
                - num_hidden_dim: Numerical hidden dimension (default: 32)
                - num_output_dim: Numerical output dimension (default: 64)
                - cat_num_categories: List[int] for embedding mode, or int for linear mode
                - cat_embedding_dim: Categorical embedding dimension (default: 16)
                - cat_output_dim: Categorical output dimension (default: 32)
                - fusion_output_dim: Fusion output dimension (default: 256)
                - fusion_dropout: Fusion dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Extract config with defaults
        num_input_dim = config.get('num_input_dim', 8)
        num_hidden_dim = config.get('num_hidden_dim', 32)
        num_output_dim = config.get('num_output_dim', 64)
        
        cat_num_categories = config.get('cat_num_categories', [2, 2, 2, 2, 2])
        cat_embedding_dim = config.get('cat_embedding_dim', 16)
        cat_output_dim = config.get('cat_output_dim', 32)
        
        fusion_output_dim = config.get('fusion_output_dim', 256)
        fusion_dropout = config.get('fusion_dropout', 0.1)
        
        # Initialize encoders
        self.numerical_encoder = NumericalEncoder(
            input_dim=num_input_dim,
            hidden_dim=num_hidden_dim,
            output_dim=num_output_dim
        )
        
        self.categorical_encoder = CategoricalEncoder(
            num_categories=cat_num_categories,
            embedding_dim=cat_embedding_dim,
            output_dim=cat_output_dim
        )
        
        self.fusion = FusionModule(
            num_dim=num_output_dim,
            cat_dim=cat_output_dim,
            output_dim=fusion_output_dim,
            dropout=fusion_dropout
        )
        
        self.output_dim = fusion_output_dim
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multi-modal features.
        
        Args:
            batch: Dictionary containing:
                - 'num_features': Tensor[batch, num_input_dim]
                - 'cat_features': Tensor[batch, cat_input_dim]
                
        Returns:
            Tensor[batch, fusion_output_dim] user embeddings
        """
        num_features = batch['num_features']
        cat_features = batch['cat_features']
        
        # Encode numerical features
        num_embed = self.numerical_encoder(num_features)
        
        # Encode categorical features
        cat_embed = self.categorical_encoder(cat_features)
        
        # Fuse embeddings
        fused = self.fusion(num_embed, cat_embed)
        
        return fused
