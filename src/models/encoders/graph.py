"""Graph encoder using Relational Graph Convolutional Networks (RGCN) for bot detection.

This module implements a graph encoder that leverages social network structure
to learn user representations through multi-layer RGCN with native support for
multiple edge types (follow, friend, mention).

RGCN is more suitable than GAT for heterogeneous social networks because:
1. Native support for multiple relation types through separate weight matrices
2. Better modeling of different semantic relationships (follow vs friend vs mention)
3. More parameter-efficient with basis decomposition
4. Proven effectiveness on Twibot-20 dataset (BotRGCN paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import PyTorch Geometric components
try:
    from torch_geometric.nn import RGCNConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    RGCNConv = None


class GraphEncoder(nn.Module):
    """图编码器: 节点特征 + 边 → 128维 (使用 RGCN)
    
    Uses Relational Graph Convolutional Network (RGCN) layers for message passing
    with native support for multiple edge types. Implements k-hop neighbor 
    aggregation through stacked RGCN layers.
    
    RGCN learns separate transformation matrices for each relation type,
    enabling better modeling of heterogeneous social network relationships.
    
    Requirements:
        - 7.1: Output 128-dimensional embedding per node
        - 7.2: Use RGCN layers for message passing (changed from GAT)
        - 7.3: Native support for multiple edge types (follow, friend, mention)
        - 7.4: Aggregate information from k-hop neighbors
        - 7.5: Apply dropout for regularization
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_relations: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_bases: Optional[int] = None,
        aggr: str = 'mean'
    ):
        """
        Args:
            input_dim: Input node feature dimension (default: 256, from other encoders)
            hidden_dim: Hidden layer dimension (default: 128)
            output_dim: Output embedding dimension (default: 128)
            num_relations: Number of relation/edge types (default: 2 for follow, friend)
            num_layers: Number of RGCN layers for k-hop aggregation (default: 2)
            dropout: Dropout probability for regularization (default: 0.1)
            num_bases: Number of bases for basis decomposition (default: None = no decomposition)
                       Using basis decomposition reduces parameters when num_relations is large
            aggr: Aggregation scheme ('mean', 'sum', 'max') (default: 'mean')
        """
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch-geometric is required for GraphEncoder. "
                "Install it with: pip install torch-geometric>=2.3.0"
            )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build RGCN layers
        self.rgcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # First layer: input_dim -> hidden_dim
            # Subsequent layers: hidden_dim -> hidden_dim
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim
            
            rgcn_layer = RGCNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                num_relations=num_relations,
                num_bases=num_bases,
                aggr=aggr
            )
            self.rgcn_layers.append(rgcn_layer)
            self.layer_norms.append(nn.LayerNorm(out_channels))
        
        # Final projection to output_dim (if different from hidden_dim)
        if hidden_dim != output_dim:
            self.output_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_projection = nn.Identity()
        
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode graph nodes using RGCN layers.
        
        Args:
            x: Tensor[n_nodes, input_dim] node features
            edge_index: Tensor[2, n_edges] edge indices (source, target)
            edge_type: Tensor[n_edges] edge types (0=follow, 1=friend, etc.)
                       If None, all edges are treated as type 0
            
        Returns:
            Tensor[n_nodes, output_dim] graph-encoded node embeddings
        """
        # Handle missing edge_type - default to all zeros (single relation type)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        
        # Apply RGCN layers for k-hop aggregation
        h = x
        for i, (rgcn_layer, layer_norm) in enumerate(zip(self.rgcn_layers, self.layer_norms)):
            # RGCN convolution with relation-specific transformations
            h = rgcn_layer(h, edge_index, edge_type)
            
            # Layer normalization
            h = layer_norm(h)
            
            # Apply activation and dropout (except for last layer before projection)
            if i < self.num_layers - 1:
                h = F.leaky_relu(h, negative_slope=0.2)
                h = self.dropout_layer(h)
        
        # Final projection
        h = self.output_projection(h)
        h = self.output_norm(h)
        
        return h
    
    def get_num_hops(self) -> int:
        """Return the number of hops (layers) for neighbor aggregation.
        
        Returns:
            Number of RGCN layers, which equals k-hop aggregation depth
        """
        return self.num_layers


class GraphEncoderFallback(nn.Module):
    """Fallback GraphEncoder when torch-geometric is not available.
    
    This is a simple MLP-based encoder that doesn't use graph structure.
    It's provided for testing and development when torch-geometric is not installed.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_relations: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_bases: Optional[int] = None,
        aggr: str = 'mean'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Simple MLP fallback (ignores graph structure)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Simple MLP forward pass (ignores graph structure)."""
        return self.mlp(x)
    
    def get_num_hops(self) -> int:
        return self.num_layers
