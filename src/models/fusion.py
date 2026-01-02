"""Multi-modal feature fusion module for bot detection model."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """注意力多模态融合: (64 + 32 + 256 + 128) → 256
    
    使用门控注意力机制和深层特征交互来融合多模态特征。
    
    架构:
    1. 各模态投影到统一维度
    2. 基于输入的动态注意力权重计算
    3. 加权融合 + 残差连接
    4. 深层MLP进行特征交互
    
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
        
        # 1. 模态投影层：将各模态投影到统一维度
        self.modality_projections = nn.ModuleDict()
        for modality in self.enabled_modalities:
            dim = self.modality_dims[modality]
            self.modality_projections[modality] = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 2. 动态注意力网络：基于输入计算注意力权重
        # 每个模态有一个注意力评分网络
        self.attention_networks = nn.ModuleDict()
        for modality in self.enabled_modalities:
            self.attention_networks[modality] = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.GELU(),
                nn.Linear(output_dim // 4, 1)
            )
        
        # 3. 深层特征交互 MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. 输出投影（带残差）
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)
        
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
        Fuse embeddings from multiple modalities using dynamic attention.
        
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
        
        # Step 1: Project each modality to common dimension
        projected = {}
        for modality, embed in available_embeddings.items():
            projected[modality] = self.modality_projections[modality](embed)
        
        # Step 2: Compute dynamic attention weights based on projected features
        attention_scores = []
        modality_order = list(projected.keys())
        
        for modality in modality_order:
            # 基于投影后的特征计算注意力分数
            score = self.attention_networks[modality](projected[modality])  # [batch, 1]
            attention_scores.append(score)
        
        # Stack and apply softmax: [batch, n_modalities]
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_probs = F.softmax(attention_scores, dim=1)
        
        # Store mean attention weights for inspection
        self._last_attention_weights = {}
        mean_weights = attention_probs.mean(dim=0)
        for i, modality in enumerate(modality_order):
            self._last_attention_weights[modality] = mean_weights[i].item()
        
        # Step 3: Weighted sum of projected embeddings
        fused = torch.zeros(batch_size, self.output_dim, device=device)
        for i, modality in enumerate(modality_order):
            weight = attention_probs[:, i:i+1]  # [batch, 1]
            fused = fused + weight * projected[modality]
        
        # Step 4: Deep feature interaction with residual connection
        interaction_out = self.interaction_mlp(fused)
        fused = fused + interaction_out  # Residual connection
        
        # Step 5: Output projection with residual
        output = self.output_projection(fused)
        output = self.output_norm(output + fused)  # Residual connection
        output = self.output_dropout(output)
        
        return output
    
    def get_attention_weights(self) -> Dict[str, float]:
        """Return the last computed attention weights for each modality.
        
        Returns:
            Dictionary mapping modality names to their attention weights.
            Weights sum to 1.0 (within numerical tolerance).
        """
        return self._last_attention_weights.copy()


class FusionModule(nn.Module):
    """多模态特征融合: (64 + 32) → 256
    
    使用深层MLP融合数值和分类特征。
    包含残差连接和多层非线性变换。
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
        
        concat_dim = num_dim + cat_dim
        
        # 深层融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 输出dropout
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
        
        # Deep fusion
        output = self.fusion_net(concat)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
