"""Prototypical Network for few-shot classification."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """原型网络
    
    Implements the prototypical network algorithm for few-shot classification.
    Computes class prototypes from support samples and classifies query samples
    based on distance to prototypes.
    
    改进:
    1. 可学习的温度参数控制softmax锐度
    2. 支持缩放点积距离（更适合高维空间）
    3. 特征归一化选项
    
    Supports multi-modal data including text and graph features.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        distance: str = 'euclidean',
        temperature: float = 1.0,
        learn_temperature: bool = True,
        normalize_features: bool = False
    ):
        """
        Args:
            encoder: Feature encoder module (e.g., MultiModalEncoder)
            distance: Distance metric - 'euclidean', 'cosine', or 'scaled_dot'
            temperature: Initial temperature for softmax scaling
            learn_temperature: Whether to learn temperature parameter
            normalize_features: Whether to L2-normalize features before distance computation
        """
        super().__init__()
        
        if distance not in ('euclidean', 'cosine', 'scaled_dot'):
            raise ValueError(f"Unknown distance metric: {distance}")
        
        self.encoder = encoder
        self.distance = distance
        self.normalize_features = normalize_features
        
        # 可学习的温度参数
        if learn_temperature:
            # 使用log scale确保温度始终为正
            self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer('log_temperature', torch.tensor(temperature).log())
    
    @property
    def temperature(self) -> torch.Tensor:
        """获取当前温度值"""
        return self.log_temperature.exp()
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类原型
        
        Computes class prototypes as the mean of support features for each class.
        
        Args:
            support_features: Tensor[n_support, embed_dim] encoded support features
            support_labels: Tensor[n_support] class labels (integers)
            
        Returns:
            Tensor[n_classes, embed_dim] class prototypes, ordered by class label
        """
        if support_features.size(0) == 0:
            raise ValueError("Support set cannot be empty")
        
        # Get unique classes sorted
        unique_classes = torch.unique(support_labels)
        unique_classes = unique_classes.sort()[0]
        n_classes = unique_classes.size(0)
        embed_dim = support_features.size(1)
        
        # Compute prototype for each class
        prototypes = torch.zeros(n_classes, embed_dim, device=support_features.device)
        
        for i, c in enumerate(unique_classes):
            mask = support_labels == c
            class_features = support_features[mask]
            prototypes[i] = class_features.mean(dim=0)
        
        return prototypes
    
    def _compute_distances(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances from queries to prototypes.
        
        Args:
            query_features: Tensor[n_query, embed_dim]
            prototypes: Tensor[n_classes, embed_dim]
            
        Returns:
            Tensor[n_query, n_classes] distances (lower = closer)
        """
        # 可选的特征归一化
        if self.normalize_features:
            query_features = F.normalize(query_features, p=2, dim=1)
            prototypes = F.normalize(prototypes, p=2, dim=1)
        
        if self.distance == 'euclidean':
            # Euclidean distance: ||q - p||^2
            diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = (diff ** 2).sum(dim=2)
            
        elif self.distance == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            query_norm = F.normalize(query_features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarity
            
        else:  # scaled_dot
            # Scaled dot product (like attention): -q·p / sqrt(d)
            # 负号是因为我们需要距离（越小越好）
            embed_dim = query_features.size(1)
            similarity = torch.mm(query_features, prototypes.t()) / (embed_dim ** 0.5)
            distances = -similarity
        
        return distances

    
    def forward(
        self,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor],
        support_texts: Optional[List[str]] = None,
        query_texts: Optional[List[str]] = None,
        support_text_embeddings: Optional[torch.Tensor] = None,
        query_text_embeddings: Optional[torch.Tensor] = None,
        support_graph_embeddings: Optional[torch.Tensor] = None,
        query_graph_embeddings: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for prototypical network.
        
        Encodes support and query samples, computes prototypes, and returns
        log-probabilities for query samples.
        
        Args:
            support_set: Dictionary containing:
                - 'num_features': Tensor[n_support, num_dim]
                - 'cat_features': Tensor[n_support, cat_dim]
                - 'labels': Tensor[n_support]
            query_set: Dictionary containing:
                - 'num_features': Tensor[n_query, num_dim]
                - 'cat_features': Tensor[n_query, cat_dim]
            support_texts: List of text strings for support samples (optional, for online encoding)
            query_texts: List of text strings for query samples (optional, for online encoding)
            support_text_embeddings: Precomputed text embeddings for support (optional)
            query_text_embeddings: Precomputed text embeddings for query (optional)
            support_graph_embeddings: Precomputed graph embeddings for support (optional)
            query_graph_embeddings: Precomputed graph embeddings for query (optional)
            edge_index: Graph edge indices (optional, for online graph encoding)
            edge_type: Edge types (optional, for online graph encoding)
            
        Note:
            - 如果同时提供 texts 和 text_embeddings，优先使用 text_embeddings
            - 如果同时提供 graph_embeddings 和 edge_index，优先使用 graph_embeddings
                
        Returns:
            Dictionary containing:
                - 'log_probs': Tensor[n_query, n_classes] log probabilities
                - 'prototypes': Tensor[n_classes, embed_dim] class prototypes
                - 'query_embeddings': Tensor[n_query, embed_dim] query embeddings
                - 'temperature': Current temperature value
        """
        # Check if encoder supports multi-modal data
        encoder_supports_multimodal = hasattr(self.encoder, 'text_encoder') or hasattr(self.encoder, 'enabled_modalities')
        
        # Encode support set
        if encoder_supports_multimodal:
            support_features = self.encoder(
                {
                    'num_features': support_set['num_features'],
                    'cat_features': support_set['cat_features']
                },
                texts=support_texts,
                text_embeddings=support_text_embeddings,
                graph_embeddings=support_graph_embeddings,
                edge_index=edge_index,
                edge_type=edge_type
            )
        else:
            # Simple encoder without multi-modal support
            support_features = self.encoder({
                'num_features': support_set['num_features'],
                'cat_features': support_set['cat_features']
            })
        support_labels = support_set['labels']
        
        # Encode query set
        if encoder_supports_multimodal:
            query_features = self.encoder(
                {
                    'num_features': query_set['num_features'],
                    'cat_features': query_set['cat_features']
                },
                texts=query_texts,
                text_embeddings=query_text_embeddings,
                graph_embeddings=query_graph_embeddings,
                edge_index=edge_index,
                edge_type=edge_type
            )
        else:
            # Simple encoder without multi-modal support
            query_features = self.encoder({
                'num_features': query_set['num_features'],
                'cat_features': query_set['cat_features']
            })
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # Compute distances
        distances = self._compute_distances(query_features, prototypes)
        
        # Convert distances to log probabilities with temperature scaling
        # 温度越高，分布越平坦；温度越低，分布越尖锐
        log_probs = F.log_softmax(-distances / self.temperature, dim=1)
        
        return {
            'log_probs': log_probs,
            'prototypes': prototypes,
            'query_embeddings': query_features,
            'temperature': self.temperature.item()
        }
    
    def classify(
        self,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor],
        support_texts: Optional[List[str]] = None,
        query_texts: Optional[List[str]] = None,
        support_text_embeddings: Optional[torch.Tensor] = None,
        query_text_embeddings: Optional[torch.Tensor] = None,
        support_graph_embeddings: Optional[torch.Tensor] = None,
        query_graph_embeddings: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify query samples given support set.
        
        Convenience method that returns predictions and probabilities.
        
        Args:
            support_set: Support set dictionary with features and labels
            query_set: Query set dictionary with features
            support_texts: List of text strings for support samples (optional)
            query_texts: List of text strings for query samples (optional)
            support_text_embeddings: Precomputed text embeddings for support (optional)
            query_text_embeddings: Precomputed text embeddings for query (optional)
            support_graph_embeddings: Precomputed graph embeddings for support (optional)
            query_graph_embeddings: Precomputed graph embeddings for query (optional)
            edge_index: Graph edge indices (optional)
            edge_type: Edge types (optional)
            
        Returns:
            Tuple of:
                - predictions: Tensor[n_query] predicted class labels
                - probabilities: Tensor[n_query, n_classes] class probabilities
        """
        output = self.forward(
            support_set, query_set,
            support_texts=support_texts,
            query_texts=query_texts,
            support_text_embeddings=support_text_embeddings,
            query_text_embeddings=query_text_embeddings,
            support_graph_embeddings=support_graph_embeddings,
            query_graph_embeddings=query_graph_embeddings,
            edge_index=edge_index,
            edge_type=edge_type
        )
        log_probs = output['log_probs']
        
        # Get predictions (argmax of log probs)
        predictions = log_probs.argmax(dim=1)
        
        # Convert log probs to probs
        probabilities = log_probs.exp()
        
        return predictions, probabilities
