"""Multi-modal encoder combining numerical, categorical, text, graph, and fusion modules."""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .encoders.numerical import NumericalEncoder
from .encoders.categorical import CategoricalEncoder
from .fusion import FusionModule, AttentionFusion


class MultiModalEncoder(nn.Module):
    """完整的多模态编码器 (支持可配置模态)
    
    Combines NumericalEncoder, CategoricalEncoder, TextEncoder, GraphEncoder,
    and FusionModule to produce unified user embeddings.
    
    支持不同数据集的分类特征维度:
    - Twibot-20: 5维二值特征 (嵌入模式)
    - Misbot: 20维one-hot特征 (线性模式)
    
    支持可配置的模态组合:
    - num: 数值特征编码器
    - cat: 分类特征编码器
    - text: 文本编码器 (XLM-RoBERTa) 或预计算嵌入
    - graph: 图编码器 (RGCN)
    
    文本模态支持两种模式:
    1. 在线编码: 使用 TextEncoder 实时编码文本
    2. 预计算嵌入: 直接使用预计算的嵌入向量 (更快)
    
    Requirements:
        - 8.1: Output 256-dimensional fused representation
        - 8.2: Support configurable modality combinations
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with keys:
                - num_input_dim: Numerical input dimension (default: 5)
                - num_hidden_dim: Numerical hidden dimension (default: 32)
                - num_output_dim: Numerical output dimension (default: 64)
                - cat_num_categories: List[int] for embedding mode, or int for linear mode
                - cat_embedding_dim: Categorical embedding dimension (default: 16)
                - cat_output_dim: Categorical output dimension (default: 32)
                - text_model_name: Pretrained model name (default: xlm-roberta-base)
                - text_output_dim: Text output dimension (default: 256)
                - text_max_length: Max token length (default: 128)
                - text_freeze_backbone: Whether to freeze text backbone (default: True)
                - use_precomputed_text_embeddings: Use precomputed embeddings (default: True)
                - graph_input_dim: Graph input dimension (default: 256)
                - graph_hidden_dim: Graph hidden dimension (default: 128)
                - graph_output_dim: Graph output dimension (default: 128)
                - graph_num_relations: Number of relation types (default: 2)
                - graph_num_layers: Number of RGCN layers (default: 2)
                - graph_dropout: Graph dropout rate (default: 0.1)
                - graph_num_bases: Number of bases for RGCN decomposition (default: None)
                - fusion_output_dim: Fusion output dimension (default: 256)
                - fusion_dropout: Fusion dropout rate (default: 0.1)
                - fusion_use_attention: Whether to use attention fusion (default: True)
                - enabled_modalities: List of enabled modalities (default: ['num', 'cat'])
        """
        super().__init__()
        
        self.config = config
        
        # Extract enabled modalities
        self.enabled_modalities = config.get('enabled_modalities', ['num', 'cat'])
        
        # Text embedding mode
        self.use_precomputed_text_embeddings = config.get('use_precomputed_text_embeddings', True)
        
        # Extract config with defaults
        num_input_dim = config.get('num_input_dim', 5)
        num_hidden_dim = config.get('num_hidden_dim', 32)
        num_output_dim = config.get('num_output_dim', 64)
        
        cat_num_categories = config.get('cat_num_categories', [2, 2, 2])
        cat_embedding_dim = config.get('cat_embedding_dim', 16)
        cat_output_dim = config.get('cat_output_dim', 32)
        
        fusion_output_dim = config.get('fusion_output_dim', 256)
        fusion_dropout = config.get('fusion_dropout', 0.1)
        fusion_use_attention = config.get('fusion_use_attention', True)
        
        # Initialize numerical encoder if enabled
        self.numerical_encoder = None
        if 'num' in self.enabled_modalities:
            self.numerical_encoder = NumericalEncoder(
                input_dim=num_input_dim,
                hidden_dim=num_hidden_dim,
                output_dim=num_output_dim
            )
        
        # Initialize categorical encoder if enabled
        self.categorical_encoder = None
        if 'cat' in self.enabled_modalities:
            self.categorical_encoder = CategoricalEncoder(
                num_categories=cat_num_categories,
                embedding_dim=cat_embedding_dim,
                output_dim=cat_output_dim
            )
        
        # Initialize text encoder if enabled (only when not using precomputed embeddings)
        self.text_encoder = None
        self.text_projection = None  # 用于预计算嵌入的投影层
        text_output_dim = config.get('text_output_dim', 256)
        text_hidden_size = config.get('text_hidden_size', 768)  # XLM-RoBERTa base hidden size
        
        if 'text' in self.enabled_modalities:
            if self.use_precomputed_text_embeddings:
                # 预计算嵌入模式: 添加可学习的投影层
                # 预计算嵌入是原始 CLS 输出 (768维)，需要投影到 text_output_dim
                self.text_projection = nn.Sequential(
                    nn.Linear(text_hidden_size, text_output_dim),
                    nn.LayerNorm(text_output_dim),
                    nn.GELU(),
                    nn.Dropout(config.get('fusion_dropout', 0.1))
                )
            else:
                # 在线编码模式: 加载完整的 TextEncoder
                from .encoders.text import TextEncoder
                
                text_model_name = config.get('text_model_name', 'xlm-roberta-base')
                text_max_length = config.get('text_max_length', 128)
                text_freeze_backbone = config.get('text_freeze_backbone', True)
                
                self.text_encoder = TextEncoder(
                    model_name=text_model_name,
                    output_dim=text_output_dim,
                    max_length=text_max_length,
                    freeze_backbone=text_freeze_backbone
                )
        
        # Initialize graph encoder if enabled
        self.graph_encoder = None
        if 'graph' in self.enabled_modalities:
            # Lazy import to avoid loading torch_geometric if not needed
            from .encoders.graph import GraphEncoder, HAS_TORCH_GEOMETRIC
            
            if HAS_TORCH_GEOMETRIC:
                graph_input_dim = config.get('graph_input_dim', 256)
                graph_hidden_dim = config.get('graph_hidden_dim', 128)
                graph_output_dim = config.get('graph_output_dim', 128)
                graph_num_relations = config.get('graph_num_relations', 2)
                graph_num_layers = config.get('graph_num_layers', 2)
                graph_dropout = config.get('graph_dropout', 0.1)
                graph_num_bases = config.get('graph_num_bases', None)
                
                self.graph_encoder = GraphEncoder(
                    input_dim=graph_input_dim,
                    hidden_dim=graph_hidden_dim,
                    output_dim=graph_output_dim,
                    num_relations=graph_num_relations,
                    num_layers=graph_num_layers,
                    dropout=graph_dropout,
                    num_bases=graph_num_bases
                )
            else:
                import warnings
                warnings.warn(
                    "torch-geometric not installed. Graph encoder will be disabled. "
                    "Install with: pip install torch-geometric>=2.3.0"
                )
                # Remove 'graph' from enabled modalities
                self.enabled_modalities = [m for m in self.enabled_modalities if m != 'graph']

        
        # Initialize fusion module
        if fusion_use_attention and len(self.enabled_modalities) > 0:
            # Use attention-based fusion for multi-modal
            graph_dim = config.get('graph_output_dim', 128) if 'graph' in self.enabled_modalities else 128
            
            self.fusion = AttentionFusion(
                num_dim=num_output_dim,
                cat_dim=cat_output_dim,
                text_dim=text_output_dim,
                graph_dim=graph_dim,
                output_dim=fusion_output_dim,
                dropout=fusion_dropout,
                enabled_modalities=self.enabled_modalities
            )
        else:
            # Use simple concatenation fusion (backward compatible)
            self.fusion = FusionModule(
                num_dim=num_output_dim,
                cat_dim=cat_output_dim,
                output_dim=fusion_output_dim,
                dropout=fusion_dropout
            )
        
        self.output_dim = fusion_output_dim
        self._fusion_use_attention = fusion_use_attention
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        texts: Optional[List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        graph_embeddings: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multi-modal features.
        
        Args:
            batch: Dictionary containing:
                - 'num_features': Tensor[batch, num_input_dim]
                - 'cat_features': Tensor[batch, cat_input_dim]
            texts: List of text strings (optional, for online text encoding)
            text_embeddings: Precomputed text embeddings Tensor[batch, text_dim] (optional)
            graph_embeddings: Precomputed graph embeddings Tensor[batch, graph_dim] (optional)
            edge_index: Graph edge indices (optional, for online graph encoding)
            edge_type: Edge types (optional, for online graph encoding)
            
        Note:
            - 如果同时提供 texts 和 text_embeddings，优先使用 text_embeddings
            - 如果同时提供 graph_embeddings 和 edge_index，优先使用 graph_embeddings
            - 使用预计算嵌入时不需要加载对应的编码器，大幅加速训练
                
        Returns:
            Tensor[batch, fusion_output_dim] user embeddings
        """
        embeddings = {}
        
        # Encode numerical features if enabled
        if self.numerical_encoder is not None and 'num_features' in batch:
            num_features = batch['num_features']
            embeddings['num'] = self.numerical_encoder(num_features)
        
        # Encode categorical features if enabled
        if self.categorical_encoder is not None and 'cat_features' in batch:
            cat_features = batch['cat_features']
            embeddings['cat'] = self.categorical_encoder(cat_features)
        
        # Handle text features
        if 'text' in self.enabled_modalities:
            if text_embeddings is not None:
                # 使用预计算的文本嵌入 (优先)
                # 预计算嵌入是原始 CLS 输出，需要通过投影层
                if self.text_projection is not None:
                    embeddings['text'] = self.text_projection(text_embeddings)
                else:
                    embeddings['text'] = text_embeddings
            elif self.text_encoder is not None and texts is not None:
                # 使用在线编码
                embeddings['text'] = self.text_encoder(texts)
        
        # Handle graph features
        if 'graph' in self.enabled_modalities:
            if graph_embeddings is not None:
                # 使用预计算的图嵌入 (优先)
                embeddings['graph'] = graph_embeddings
            elif self.graph_encoder is not None and edge_index is not None:
                # 使用在线图编码（需要所有节点的特征）
                if embeddings:
                    available_embeds = list(embeddings.values())
                    graph_input = torch.cat(available_embeds, dim=1)
                else:
                    batch_size = batch.get('num_features', batch.get('cat_features')).size(0)
                    device = next(self.parameters()).device
                    graph_input = torch.zeros(batch_size, self.graph_encoder.input_dim, device=device)
                
                embeddings['graph'] = self.graph_encoder(graph_input, edge_index, edge_type)

        
        # Fuse embeddings
        if self._fusion_use_attention:
            # Use attention fusion with keyword arguments
            fused = self.fusion(
                num_embed=embeddings.get('num'),
                cat_embed=embeddings.get('cat'),
                text_embed=embeddings.get('text'),
                graph_embed=embeddings.get('graph')
            )
        else:
            # Use simple fusion (backward compatible)
            num_embed = embeddings.get('num')
            cat_embed = embeddings.get('cat')
            if num_embed is None or cat_embed is None:
                raise ValueError("Simple fusion requires both num and cat embeddings")
            fused = self.fusion(num_embed, cat_embed)
        
        return fused
    
    def freeze_text_backbone(self) -> None:
        """Freeze text encoder backbone weights.
        
        Only the projection layer will be trained.
        """
        if self.text_encoder is not None:
            self.text_encoder.freeze_backbone()
    
    def unfreeze_text_backbone(self) -> None:
        """Unfreeze text encoder backbone weights.
        
        All text encoder parameters will be trained.
        """
        if self.text_encoder is not None:
            self.text_encoder.unfreeze_backbone()
    
    def is_text_backbone_frozen(self) -> bool:
        """Check if text encoder backbone is frozen.
        
        Returns:
            True if text backbone is frozen, False otherwise.
            Returns True if text encoder is not enabled.
        """
        if self.text_encoder is not None:
            return self.text_encoder.is_backbone_frozen
        return True
    
    def get_attention_weights(self) -> Optional[Dict[str, float]]:
        """Get attention weights from fusion module.
        
        Returns:
            Dictionary of modality attention weights if using attention fusion,
            None otherwise.
        """
        if self._fusion_use_attention and hasattr(self.fusion, 'get_attention_weights'):
            return self.fusion.get_attention_weights()
        return None
