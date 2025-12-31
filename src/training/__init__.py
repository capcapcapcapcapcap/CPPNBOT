# Training module for meta-learning training and evaluation
"""
Training module for prototypical network meta-learning.

This module provides:
- MetaTrainer: Episodic meta-learning trainer with early stopping
- Evaluator: Few-shot evaluation with multiple K-shot support
"""

from .meta_trainer import MetaTrainer
from .evaluator import Evaluator

__all__ = ['MetaTrainer', 'Evaluator']
