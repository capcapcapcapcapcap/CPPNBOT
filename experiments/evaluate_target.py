#!/usr/bin/env python
"""
Cross-Domain Evaluation Script

Evaluates a pre-trained prototypical network on the target domain (Misbot)
using few-shot adaptation with different K-shot values.

Supports multi-modal evaluation with:
- Numerical and categorical features (baseline)
- Text encoding with XLM-RoBERTa
- Graph encoding with GAT

Implements Requirements 13.2
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import BotDataset, EpisodeSampler
from src.models import MultiModalEncoder, PrototypicalNetwork
from src.training import Evaluator


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging to console and file."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler - 只输出消息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        description="Evaluate prototypical network on target domain (Misbot)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint (default: {output_dir}/best_model.pt)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="misbot",
        help="Target dataset name (default: misbot)"
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K-shot values to evaluate (default: 1 5 10 20)"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes per K-shot evaluation (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
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
        help="Enabled modalities (overrides checkpoint config). Options: num, cat, text, graph"
    )
    return parser.parse_args()


def load_model(
    model_path: str,
    config,
    device: torch.device,
    modalities_override: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> PrototypicalNetwork:
    """Load pre-trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        device: Device to load model on
        modalities_override: Override modalities from checkpoint
        logger: Logger instance
        
    Returns:
        Loaded PrototypicalNetwork model
    """
    # Load checkpoint to get saved configuration
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get enabled modalities from checkpoint or override
    if modalities_override:
        enabled_modalities = modalities_override
        if logger:
            logger.info(f"Using override modalities: {', '.join(enabled_modalities)}")
    elif 'enabled_modalities' in checkpoint:
        enabled_modalities = checkpoint['enabled_modalities']
        if logger:
            logger.info(f"Using checkpoint modalities: {', '.join(enabled_modalities)}")
    else:
        enabled_modalities = config.model.enabled_modalities
        if logger:
            logger.info(f"Using config modalities: {', '.join(enabled_modalities)}")
    
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
            'text_max_length': config.model.text_max_length,
            'text_freeze_backbone': True,  # Always freeze during evaluation
        })
    
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
    
    # Initialize encoder and model
    encoder = MultiModalEncoder(model_config)
    model = PrototypicalNetwork(encoder, distance=config.model.distance_metric)
    
    # Load checkpoint weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model, enabled_modalities


def evaluate_k_shot(
    model: PrototypicalNetwork,
    dataset: BotDataset,
    k_shot: int,
    n_episodes: int,
    device: torch.device,
    logger: logging.Logger,
    enabled_modalities: List[str]
) -> Dict[str, float]:
    """Evaluate model with specific K-shot value."""
    # Get indices
    train_indices = dataset.get_split_indices('train')
    test_indices = dataset.get_split_indices('test')
    
    # Create evaluator with modality configuration
    evaluator = Evaluator(model, enabled_modalities=enabled_modalities)
    
    # Evaluate
    metrics = evaluator.evaluate_with_k_shot(
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        k_shot=k_shot,
        n_episodes=n_episodes
    )
    
    logger.info(f"{k_shot:2d}-shot | Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed:
        config.seed = args.seed
    
    # 确定模型路径：优先命令行参数，否则用 output_dir/best_model.pt
    output_dir = Path(config.output_dir)
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = output_dir / "best_model.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print(f"Please train first or specify --model-path")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging (简洁格式)
    logger = setup_logging(output_dir)
    logger.info(f"Evaluating: {model_path}")
    
    # Set random seed (None = 随机)
    actual_seed = set_seed(config.seed)
    logger.info(f"Seed: {actual_seed}")
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model with modality configuration
    model, enabled_modalities = load_model(
        str(model_path), 
        config, 
        device,
        modalities_override=args.modalities,
        logger=logger
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params, device={device}")
    logger.info(f"Modalities: {', '.join(enabled_modalities)}")
    
    # Load target dataset
    dataset = BotDataset(args.dataset, config.data_dir)
    logger.info(f"Target: {args.dataset} ({len(dataset)} users)")
    
    # Evaluation configuration
    logger.info(f"K-shots: {args.k_shots}, Episodes: {args.n_episodes}")
    
    # Evaluate with different K-shot values
    results = {}
    for k_shot in args.k_shots:
        metrics = evaluate_k_shot(
            model=model,
            dataset=dataset,
            k_shot=k_shot,
            n_episodes=args.n_episodes,
            device=device,
            logger=logger,
            enabled_modalities=enabled_modalities
        )
        results[k_shot] = metrics
    
    # Save results
    results_path = output_dir / f"eval_{args.dataset}.json"
    
    # Convert keys to strings for JSON serialization
    json_results = {
        'dataset': args.dataset,
        'model_path': str(model_path),
        'n_episodes': args.n_episodes,
        'enabled_modalities': enabled_modalities,
        'results': {str(k): v for k, v in results.items()}
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved: {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
