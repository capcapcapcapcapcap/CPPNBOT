"""
Configuration management module for prototypical network experiments.

This module provides dataclasses for model, training, and experiment configuration,
along with utilities for loading configurations from YAML files.

Supports multi-modal configuration including:
- Numerical encoder
- Categorical encoder
- Text encoder (XLM-RoBERTa)
- Graph encoder (GAT)
- Attention-based fusion
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Supports multi-modal encoders and configurable modality combinations.
    """
    
    # Numerical encoder
    num_input_dim: int = 5
    num_hidden_dim: int = 32
    num_output_dim: int = 64
    
    # Categorical encoder
    # 可以是 List[int] (嵌入模式) 或 int (线性模式)
    # Twibot-20: [2, 2, 2] (3个二值特征)
    # Misbot: [2, 2, 2] (前3维)
    cat_num_categories: List[int] = field(default_factory=lambda: [2, 2, 2])
    cat_embedding_dim: int = 16
    cat_output_dim: int = 32
    
    # Text encoder (XLM-RoBERTa)
    text_model_name: str = "xlm-roberta-base"
    text_output_dim: int = 256
    text_max_length: int = 512
    text_freeze_backbone: bool = True
    
    # Graph encoder (GAT)
    graph_input_dim: int = 256
    graph_hidden_dim: int = 128
    graph_output_dim: int = 128
    graph_num_heads: int = 4
    graph_num_layers: int = 2
    graph_dropout: float = 0.1
    
    # Fusion module
    fusion_output_dim: int = 256
    fusion_dropout: float = 0.1
    fusion_use_attention: bool = True
    
    # Enabled modalities: ['num', 'cat', 'text', 'graph']
    enabled_modalities: List[str] = field(default_factory=lambda: ['num', 'cat'])
    
    # Distance metric
    distance_metric: str = 'euclidean'
    
    def __post_init__(self):
        """Validate model configuration parameters."""
        if self.num_input_dim <= 0:
            raise ValueError(f"num_input_dim must be positive, got {self.num_input_dim}")
        if self.num_hidden_dim <= 0:
            raise ValueError(f"num_hidden_dim must be positive, got {self.num_hidden_dim}")
        if self.num_output_dim <= 0:
            raise ValueError(f"num_output_dim must be positive, got {self.num_output_dim}")
        if self.cat_embedding_dim <= 0:
            raise ValueError(f"cat_embedding_dim must be positive, got {self.cat_embedding_dim}")
        if self.cat_output_dim <= 0:
            raise ValueError(f"cat_output_dim must be positive, got {self.cat_output_dim}")
        if self.text_output_dim <= 0:
            raise ValueError(f"text_output_dim must be positive, got {self.text_output_dim}")
        if self.text_max_length <= 0:
            raise ValueError(f"text_max_length must be positive, got {self.text_max_length}")
        if self.graph_input_dim <= 0:
            raise ValueError(f"graph_input_dim must be positive, got {self.graph_input_dim}")
        if self.graph_hidden_dim <= 0:
            raise ValueError(f"graph_hidden_dim must be positive, got {self.graph_hidden_dim}")
        if self.graph_output_dim <= 0:
            raise ValueError(f"graph_output_dim must be positive, got {self.graph_output_dim}")
        if self.graph_num_heads <= 0:
            raise ValueError(f"graph_num_heads must be positive, got {self.graph_num_heads}")
        if self.graph_num_layers <= 0:
            raise ValueError(f"graph_num_layers must be positive, got {self.graph_num_layers}")
        if not 0.0 <= self.graph_dropout < 1.0:
            raise ValueError(f"graph_dropout must be in [0, 1), got {self.graph_dropout}")
        if self.fusion_output_dim <= 0:
            raise ValueError(f"fusion_output_dim must be positive, got {self.fusion_output_dim}")
        if not 0.0 <= self.fusion_dropout < 1.0:
            raise ValueError(f"fusion_dropout must be in [0, 1), got {self.fusion_dropout}")
        if self.distance_metric not in ('euclidean', 'cosine'):
            raise ValueError(f"distance_metric must be 'euclidean' or 'cosine', got {self.distance_metric}")
        if not self.cat_num_categories or any(n <= 0 for n in self.cat_num_categories):
            raise ValueError(f"cat_num_categories must be non-empty with positive values, got {self.cat_num_categories}")
        
        # Validate enabled_modalities
        valid_modalities = {'num', 'cat', 'text', 'graph'}
        for modality in self.enabled_modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality '{modality}'. Must be one of {valid_modalities}")


@dataclass
class TrainingConfig:
    """Training configuration for meta-learning.
    
    Supports multi-modal training with separate learning rates for text encoder.
    """
    
    # Episode configuration
    n_way: int = 2
    k_shot: int = 5
    n_query: int = 15
    
    # Training episodes
    n_episodes_train: int = 100
    n_episodes_val: int = 50
    
    # Training parameters
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Text encoder learning rate (typically smaller for fine-tuning)
    text_learning_rate: float = 1e-5
    
    # Early stopping
    patience: int = 10
    
    def __post_init__(self):
        """Validate training configuration parameters."""
        if self.n_way <= 0:
            raise ValueError(f"n_way must be positive, got {self.n_way}")
        if self.k_shot <= 0:
            raise ValueError(f"k_shot must be positive, got {self.k_shot}")
        if self.n_query <= 0:
            raise ValueError(f"n_query must be positive, got {self.n_query}")
        if self.n_episodes_train <= 0:
            raise ValueError(f"n_episodes_train must be positive, got {self.n_episodes_train}")
        if self.n_episodes_val <= 0:
            raise ValueError(f"n_episodes_val must be positive, got {self.n_episodes_val}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.text_learning_rate <= 0:
            raise ValueError(f"text_learning_rate must be positive, got {self.text_learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")


@dataclass
class Config:
    """Complete experiment configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data paths
    data_dir: str = "processed_data"
    output_dir: str = "results"
    
    # Reproducibility (None = 随机种子)
    seed: int = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


def _dict_to_model_config(d: dict) -> ModelConfig:
    """Convert dictionary to ModelConfig."""
    return ModelConfig(
        num_input_dim=d.get('num_input_dim', 5),
        num_hidden_dim=d.get('num_hidden_dim', 32),
        num_output_dim=d.get('num_output_dim', 64),
        cat_num_categories=d.get('cat_num_categories', [2, 2, 2]),
        cat_embedding_dim=d.get('cat_embedding_dim', 16),
        cat_output_dim=d.get('cat_output_dim', 32),
        text_model_name=d.get('text_model_name', 'xlm-roberta-base'),
        text_output_dim=d.get('text_output_dim', 256),
        text_max_length=d.get('text_max_length', 512),
        text_freeze_backbone=d.get('text_freeze_backbone', True),
        graph_input_dim=d.get('graph_input_dim', 256),
        graph_hidden_dim=d.get('graph_hidden_dim', 128),
        graph_output_dim=d.get('graph_output_dim', 128),
        graph_num_heads=d.get('graph_num_heads', 4),
        graph_num_layers=d.get('graph_num_layers', 2),
        graph_dropout=d.get('graph_dropout', 0.1),
        fusion_output_dim=d.get('fusion_output_dim', 256),
        fusion_dropout=d.get('fusion_dropout', 0.1),
        fusion_use_attention=d.get('fusion_use_attention', True),
        enabled_modalities=d.get('enabled_modalities', ['num', 'cat']),
        distance_metric=d.get('distance_metric', 'euclidean'),
    )


def _dict_to_training_config(d: dict) -> TrainingConfig:
    """Convert dictionary to TrainingConfig."""
    return TrainingConfig(
        n_way=d.get('n_way', 2),
        k_shot=d.get('k_shot', 5),
        n_query=d.get('n_query', 15),
        n_episodes_train=d.get('n_episodes_train', 100),
        n_episodes_val=d.get('n_episodes_val', 50),
        n_epochs=d.get('n_epochs', 100),
        learning_rate=d.get('learning_rate', 1e-3),
        weight_decay=d.get('weight_decay', 1e-4),
        text_learning_rate=d.get('text_learning_rate', 1e-5),
        patience=d.get('patience', 10),
    )


def load_config(path: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        Config object with loaded parameters.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML syntax is invalid.
        ValueError: If parameter values are invalid.
        KeyError: If required parameters are missing.
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML syntax in config: {e}")
    
    if data is None:
        data = {}
    
    # Parse model config
    model_data = data.get('model', {})
    model_config = _dict_to_model_config(model_data)
    
    # Parse training config
    training_data = data.get('training', {})
    training_config = _dict_to_training_config(training_data)
    
    # Create main config
    config = Config(
        model=model_config,
        training=training_config,
        data_dir=data.get('data_dir', 'processed_data'),
        output_dir=data.get('output_dir', 'results'),
        seed=data.get('seed', 42),
    )
    
    return config
