#!/usr/bin/env python
"""
Source Domain Training Script

Trains a prototypical network on the source domain (Twibot-20) for
cross-domain bot detection.

Implements Requirements 10.1, 10.3, 10.4
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

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
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train prototypical network on source domain (Twibot-20)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
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
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.training.n_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.seed:
        config.seed = args.seed
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("Prototypical Network Training - Source Domain")
    logger.info("=" * 60)
    
    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = BotDataset(args.dataset, config.data_dir)
    logger.info(f"Dataset size: {len(dataset)} users")
    logger.info(f"Train samples: {len(dataset.get_split_indices('train'))}")
    logger.info(f"Val samples: {len(dataset.get_split_indices('val'))}")
    logger.info(f"Test samples: {len(dataset.get_split_indices('test'))}")
    
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
    }
    
    # Initialize encoder and model
    logger.info("Initializing model...")
    encoder = MultiModalEncoder(model_config)
    model = PrototypicalNetwork(encoder, distance=config.model.distance_metric)
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
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
    }
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = MetaTrainer(model, dataset, trainer_config)
    
    # Log training configuration
    logger.info("Training configuration:")
    logger.info(f"  N-way: {config.training.n_way}")
    logger.info(f"  K-shot: {config.training.k_shot}")
    logger.info(f"  N-query: {config.training.n_query}")
    logger.info(f"  Episodes per epoch: {config.training.n_episodes_train}")
    logger.info(f"  Validation episodes: {config.training.n_episodes_val}")
    logger.info(f"  Max epochs: {config.training.n_epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Weight decay: {config.training.weight_decay}")
    logger.info(f"  Early stopping patience: {config.training.patience}")
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Log final results
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final training accuracy: {history['train_accuracy'][-1]:.4f}")
    logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
