#!/usr/bin/env python
"""
Source Domain Training Script

Trains a prototypical network on the source domain (Twibot-20) for
cross-domain bot detection.

Supports multi-modal training with:
- Numerical and categorical features (baseline)
- Text encoding with XLM-RoBERTa
- Graph encoding with GAT
- Ablation experiments with different modality combinations

Implements Requirements 13.1, 13.5
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config
from src.data import BotDataset
from src.models import MultiModalEncoder, PrototypicalNetwork
from src.training import MetaTrainer


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging to console and file."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler - 只输出消息内容
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - 固定名称 train.log
    log_file = output_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = None) -> int:
    """Set random seeds. If seed is None, use random seed."""
    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train prototypical network on source domain (Twibot-20)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/ablation_all.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="twibot20",
        help="Source dataset name (default: twibot20)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detect if not specified)"
    )
    # Multi-modal arguments
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=None,
        help="Enabled modalities (overrides config). Options: num, cat, text, graph"
    )
    parser.add_argument(
        "--freeze-text",
        action="store_true",
        default=None,
        help="Freeze text encoder backbone (overrides config)"
    )
    parser.add_argument(
        "--unfreeze-text",
        action="store_true",
        default=False,
        help="Unfreeze text encoder backbone (overrides config)"
    )
    parser.add_argument(
        "--text-lr",
        type=float,
        default=None,
        help="Text encoder learning rate (overrides config)"
    )
    # Ablation experiment shortcuts
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["baseline", "text", "graph", "text_graph", "all"],
        default=None,
        help="Ablation experiment preset: baseline (num+cat), text (num+cat+text), "
             "graph (num+cat+graph), text_graph (num+cat+text+graph), all (same as text_graph)"
    )
    return parser.parse_args()


def get_modalities_from_ablation(ablation: str) -> List[str]:
    """Get modality list from ablation preset name."""
    presets = {
        "baseline": ["num", "cat"],
        "text": ["num", "cat", "text"],
        "graph": ["num", "cat", "graph"],
        "text_graph": ["num", "cat", "text", "graph"],
        "all": ["num", "cat", "text", "graph"]
    }
    return presets.get(ablation, ["num", "cat"])


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.epochs:
        config.training.n_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.seed is not None:
        config.seed = args.seed
    
    # Handle modality configuration
    if args.ablation:
        # Ablation preset takes precedence
        config.model.enabled_modalities = get_modalities_from_ablation(args.ablation)
    elif args.modalities:
        # Explicit modality list
        config.model.enabled_modalities = args.modalities
    
    # Handle text backbone freezing
    if args.unfreeze_text:
        config.model.text_freeze_backbone = False
    elif args.freeze_text:
        config.model.text_freeze_backbone = True
    
    # Handle text learning rate
    if args.text_lr:
        config.training.text_learning_rate = args.text_lr
    
    # 自动生成带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Include modality info in output directory name
    modality_suffix = "_".join(config.model.enabled_modalities)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.output_dir) / f"{timestamp}_{modality_suffix}"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Prototypical Network Training")
    
    # Set random seed (None = 随机)
    actual_seed = set_seed(config.seed)
    logger.info(f"Seed: {actual_seed}")
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = BotDataset(args.dataset, config.data_dir)
    logger.info(f"Dataset: {args.dataset} ({len(dataset)} users, "
                f"train={len(dataset.get_split_indices('train'))}, "
                f"val={len(dataset.get_split_indices('val'))})")
    
    # Log enabled modalities
    enabled_modalities = config.model.enabled_modalities
    logger.info(f"Modalities: {', '.join(enabled_modalities)}")
    
    # Create model configuration dict
    model_config = {
        'num_input_dim': config.model.num_input_dim,
        'num_hidden_dim': config.model.num_hidden_dim,
        'num_output_dim': config.model.num_output_dim,
        'cat_num_categories': config.model.cat_num_categories,
        'cat_embedding_dim': config.model.cat_embedding_dim,
        'cat_output_dim': config.model.cat_output_dim,
        'fusion_output_dim': config.model.fusion_output_dim,
        'fusion_dropout': config.model.fusion_dropout,
        'fusion_use_attention': config.model.fusion_use_attention,
        'enabled_modalities': enabled_modalities,
    }
    
    # Add text encoder config if text modality is enabled
    if 'text' in enabled_modalities:
        model_config.update({
            'text_model_name': config.model.text_model_name,
            'text_output_dim': config.model.text_output_dim,
            'text_hidden_size': config.model.text_hidden_size,
            'text_max_length': config.model.text_max_length,
            'text_freeze_backbone': config.model.text_freeze_backbone,
        })
        logger.info(f"Text encoder: {config.model.text_model_name}, "
                   f"freeze_backbone={config.model.text_freeze_backbone}")
    
    # Add graph encoder config if graph modality is enabled
    if 'graph' in enabled_modalities:
        model_config.update({
            'graph_input_dim': config.model.graph_input_dim,
            'graph_hidden_dim': config.model.graph_hidden_dim,
            'graph_output_dim': config.model.graph_output_dim,
            'graph_num_relations': config.model.graph_num_relations,
            'graph_num_layers': config.model.graph_num_layers,
            'graph_dropout': config.model.graph_dropout,
            'graph_num_bases': config.model.graph_num_bases,
        })
        logger.info(f"Graph encoder: {config.model.graph_num_layers} RGCN layers, "
                   f"{config.model.graph_num_relations} relation types")
    
    # Initialize encoder and model
    encoder = MultiModalEncoder(model_config)
    model = PrototypicalNetwork(
        encoder, 
        distance=config.model.distance_metric,
        temperature=config.model.proto_temperature,
        learn_temperature=config.model.proto_learn_temperature,
        normalize_features=config.model.proto_normalize_features
    )
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable), device={device}")
    
    # Create trainer configuration dict
    trainer_config = {
        'n_way': config.training.n_way,
        'k_shot': config.training.k_shot,
        'n_query': config.training.n_query,
        'n_episodes_train': config.training.n_episodes_train,
        'n_episodes_val': config.training.n_episodes_val,
        'n_epochs': config.training.n_epochs,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'patience': config.training.patience,
        'output_dir': str(output_dir),
        'enabled_modalities': enabled_modalities,
        'text_learning_rate': config.training.text_learning_rate,
        'text_freeze_backbone': config.model.text_freeze_backbone,
    }
    
    # Initialize trainer
    trainer = MetaTrainer(model, dataset, trainer_config)
    
    # Log training configuration (简洁版)
    logger.info(f"Config: {config.training.n_way}-way {config.training.k_shot}-shot, "
                f"lr={config.training.learning_rate}, patience={config.training.patience}")
    
    if 'text' in enabled_modalities:
        logger.info(f"Text LR: {config.training.text_learning_rate}")
    
    # Train model
    logger.info("Training...")
    history = trainer.train()
    
    # Log final results (简洁版)
    logger.info("=" * 60)
    logger.info(f"Best val loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Final - Acc: {history['val_accuracy'][-1]:.4f}, "
                f"F1: {history['val_f1'][-1]:.4f}, "
                f"P: {history['val_precision'][-1]:.4f}, "
                f"R: {history['val_recall'][-1]:.4f}")
    logger.info(f"Model: {output_dir / 'best_model.pt'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
