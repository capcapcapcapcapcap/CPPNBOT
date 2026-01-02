#!/usr/bin/env python
"""
带微调的跨域评估脚本

在目标域评估时，先用 support set 微调模型，然后再预测。
对比原型网络的"只换原型"方式，微调能更好地适应域差异。

使用方法:
    python experiments/evaluate_with_finetune.py -m results/xxx/best_model.pt --dataset misbot
    python experiments/evaluate_with_finetune.py -m results/xxx/best_model.pt --finetune-steps 20
    python experiments/evaluate_with_finetune.py -m results/xxx/best_model.pt --finetune-mode head
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import BotDataset
from src.models import MultiModalEncoder, PrototypicalNetwork
from src.training.finetune_evaluator import FinetuneEvaluator
from src.training.evaluator import Evaluator


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    
    log_file = output_dir / f"eval_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


def load_model(model_path: str, config, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    
    enabled_modalities = checkpoint.get('enabled_modalities', config.model.enabled_modalities)
    
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
    
    if 'text' in enabled_modalities:
        model_config.update({
            'text_model_name': config.model.text_model_name,
            'text_output_dim': config.model.text_output_dim,
            'text_hidden_size': config.model.text_hidden_size,
            'text_max_length': config.model.text_max_length,
            'text_freeze_backbone': True,
        })
    
    if 'graph' in enabled_modalities:
        model_config.update({
            'graph_input_dim': config.model.graph_input_dim,
            'graph_hidden_dim': config.model.graph_hidden_dim,
            'graph_output_dim': config.model.graph_output_dim,
            'graph_num_relations': config.model.graph_num_relations,
            'graph_num_layers': config.model.graph_num_layers,
            'graph_dropout': config.model.graph_dropout,
        })
    
    encoder = MultiModalEncoder(model_config)
    model = PrototypicalNetwork(encoder, distance=config.model.distance_metric)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, enabled_modalities


def parse_args():
    parser = argparse.ArgumentParser(description="带微调的跨域评估")
    parser.add_argument("--config", "-c", type=str, default="configs/ablation_all.yaml")
    parser.add_argument("--model-path", "-m", type=str, required=True, help="模型路径")
    parser.add_argument("--dataset", type=str, default="misbot", help="目标数据集")
    parser.add_argument("--k-shots", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--finetune-steps", type=int, default=10, help="微调步数")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="微调学习率")
    parser.add_argument("--finetune-mode", type=str, default="full", choices=["full", "head"],
                       help="微调模式: full=全模型, head=只微调分类头")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compare-baseline", action="store_true", help="同时运行原型网络基线对比")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("带微调的跨域评估")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Target: {args.dataset}")
    logger.info(f"Finetune: {args.finetune_steps} steps, lr={args.finetune_lr}, mode={args.finetune_mode}")
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    model, enabled_modalities = load_model(str(model_path), config, device)
    logger.info(f"Modalities: {', '.join(enabled_modalities)}")
    
    dataset = BotDataset(args.dataset, config.data_dir)
    train_indices = dataset.get_split_indices('train')
    test_indices = dataset.get_split_indices('test')
    logger.info(f"Dataset: {len(dataset)} users, train={len(train_indices)}, test={len(test_indices)}")
    
    # 原型网络基线（可选）
    if args.compare_baseline:
        logger.info("\n--- Baseline (Prototypical Network, no finetune) ---")
        baseline_evaluator = Evaluator(model, enabled_modalities=enabled_modalities)
        baseline_results = baseline_evaluator.evaluate_multiple_k_shots(
            dataset, train_indices, test_indices,
            k_shots=args.k_shots, n_episodes=args.n_episodes
        )
        for k, metrics in baseline_results.items():
            logger.info(f"{k:2d}-shot | Acc: {metrics['accuracy']:.4f} | "
                       f"F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
    
    # 带微调的评估
    logger.info(f"\n--- With Finetune ({args.finetune_steps} steps, {args.finetune_mode} mode) ---")
    ft_evaluator = FinetuneEvaluator(
        model,
        enabled_modalities=enabled_modalities,
        finetune_steps=args.finetune_steps,
        finetune_lr=args.finetune_lr,
        finetune_mode=args.finetune_mode
    )
    
    ft_results = ft_evaluator.evaluate_multiple_k_shots(
        dataset, train_indices, test_indices,
        k_shots=args.k_shots, n_episodes=args.n_episodes
    )
    
    # 保存结果
    results = {
        'dataset': args.dataset,
        'model_path': str(model_path),
        'finetune_steps': args.finetune_steps,
        'finetune_lr': args.finetune_lr,
        'finetune_mode': args.finetune_mode,
        'n_episodes': args.n_episodes,
        'enabled_modalities': enabled_modalities,
        'results': {str(k): v for k, v in ft_results.items()}
    }
    
    if args.compare_baseline:
        results['baseline_results'] = {str(k): v for k, v in baseline_results.items()}
    
    results_path = output_dir / f"eval_finetune_{args.dataset}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
