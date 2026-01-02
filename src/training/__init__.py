# Training module for meta-learning training and evaluation
"""
Training module for prototypical network meta-learning.

This module provides:
- MetaTrainer: Episodic meta-learning trainer with early stopping
- Evaluator: Few-shot evaluation with multiple K-shot support
- FinetuneEvaluator: Few-shot evaluation with support set finetuning
"""

from .meta_trainer import MetaTrainer
from .evaluator import Evaluator
from .finetune_evaluator import FinetuneEvaluator

__all__ = ['MetaTrainer', 'Evaluator', 'FinetuneEvaluator']
