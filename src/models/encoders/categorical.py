"""Categorical feature encoder for bot detection model."""

from typing import List, Union

import torch
import torch.nn as nn


class CategoricalEncoder(nn.Module):
    """分类特征编码器: D维 → output_dim维
    
    支持两种模式:
    1. 嵌入模式 (num_categories为列表): 每个特征使用独立的嵌入表
    2. 线性模式 (num_categories为整数): 直接使用线性层投影
    
    这样可以支持:
    - Twibot-20: 5维二值特征 → 32维
    - Misbot: 20维one-hot特征 → 32维
    """
    
    def __init__(
        self,
        num_categories: Union[List[int], int],
        embedding_dim: int = 16,
        output_dim: int = 32
    ):
        """
        Args:
            num_categories: 
                - List[int]: 每个特征的类别数 (嵌入模式)
                - int: 输入特征维度 (线性模式，用于one-hot输入)
            embedding_dim: 每个特征的嵌入维度 (仅嵌入模式使用)
            output_dim: 输出嵌入维度
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        
        if isinstance(num_categories, int):
            # 线性模式: 直接投影 (适用于Misbot的20维one-hot)
            self.mode = 'linear'
            self.input_dim = num_categories
            self.projection = nn.Sequential(
                nn.Linear(num_categories, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim)
            )
        else:
            # 嵌入模式: 每个特征独立嵌入 (适用于Twibot-20的5维二值特征)
            self.mode = 'embedding'
            self.num_categories = num_categories
            self.num_features = len(num_categories)
            self.input_dim = self.num_features
            
            # Create separate embedding table for each categorical feature
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_cat, embedding_dim)
                for num_cat in num_categories
            ])
            
            # Projection layer: concatenated embeddings → output_dim
            concat_dim = self.num_features * embedding_dim
            self.projection = nn.Sequential(
                nn.Linear(concat_dim, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode categorical features.
        
        Args:
            x: Tensor[batch, input_dim] categorical features
               - 嵌入模式: 整数索引
               - 线性模式: 浮点数 (one-hot或二值)
            
        Returns:
            Tensor[batch, output_dim] encoded embeddings
        """
        if self.mode == 'linear':
            # 线性模式: 直接投影
            return self.projection(x.float())
        else:
            # 嵌入模式: 查表 + 拼接 + 投影
            x = x.long()  # 确保是整数索引
            embedded = []
            for i, emb in enumerate(self.embeddings):
                embedded.append(emb(x[:, i]))
            
            concat = torch.cat(embedded, dim=1)
            return self.projection(concat)
