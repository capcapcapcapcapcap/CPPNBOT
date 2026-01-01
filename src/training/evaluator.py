"""
Evaluator: 少样本评估器

Implements Requirements 8.1, 8.2, 8.3, 8.4
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..data.dataset import BotDataset
from ..data.episode_sampler import EpisodeSampler
from ..models.prototypical import PrototypicalNetwork


class Evaluator:
    """少样本评估器
    
    Evaluates prototypical network performance using few-shot adaptation.
    Supports evaluation with different K-shot values and computes
    accuracy, precision, recall, and F1 score.
    """
    
    def __init__(self, model: PrototypicalNetwork):
        """
        Initialize the evaluator.
        
        Args:
            model: PrototypicalNetwork instance (should be pre-trained)
        """
        self.model = model
        
        # Device
        self.device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    def _move_to_device(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move data dictionary to the model's device."""
        return {k: v.to(self.device) for k, v in data.items()}
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        positive_class: int = 1
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            positive_class: Label for the positive class (bot = 1)
            
        Returns:
            Dictionary with accuracy, precision, recall, and F1 score
        """
        predictions = predictions.cpu()
        labels = labels.cpu()
        
        # Compute confusion matrix components
        # True Positives: predicted positive and actually positive
        tp = ((predictions == positive_class) & (labels == positive_class)).sum().item()
        # True Negatives: predicted negative and actually negative
        tn = ((predictions != positive_class) & (labels != positive_class)).sum().item()
        # False Positives: predicted positive but actually negative
        fp = ((predictions == positive_class) & (labels != positive_class)).sum().item()
        # False Negatives: predicted negative but actually positive
        fn = ((predictions != positive_class) & (labels == positive_class)).sum().item()
        
        # Compute metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }

    def few_shot_evaluate(
        self,
        support_set: Dict[str, torch.Tensor],
        test_dataset: BotDataset,
        test_indices: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform few-shot evaluation.
        
        Uses the provided support set to compute prototypes and classifies
        all samples in the test set.
        
        Args:
            support_set: Dictionary containing:
                - 'num_features': Tensor[n_support, num_dim]
                - 'cat_features': Tensor[n_support, cat_dim]
                - 'labels': Tensor[n_support]
            test_dataset: BotDataset for test samples
            test_indices: Indices of test samples to evaluate
            
        Returns:
            Dictionary with accuracy, precision, recall, and F1 score
        """
        self.model.eval()
        
        # Move support set to device
        support_set = self._move_to_device(support_set)
        
        # Collect all test samples
        test_num_features = []
        test_cat_features = []
        test_labels = []
        
        for idx in test_indices:
            item = test_dataset[idx.item()]
            test_num_features.append(item['num_features'])
            test_cat_features.append(item['cat_features'])
            test_labels.append(item['label'])
        
        # Stack into tensors (cat_features 需要转为 long)
        query_set = {
            'num_features': torch.stack(test_num_features).to(self.device),
            'cat_features': torch.stack(test_cat_features).long().to(self.device)
        }
        test_labels = torch.stack(test_labels).to(self.device)
        
        # Forward pass with frozen encoder
        with torch.no_grad():
            output = self.model(support_set, query_set)
            log_probs = output['log_probs']
            predictions = log_probs.argmax(dim=1)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, test_labels)
        
        return metrics
    
    def evaluate_with_k_shot(
        self,
        dataset: BotDataset,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        k_shot: int,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate with a specific K-shot value.
        
        Samples multiple support sets and averages the results.
        
        Args:
            dataset: BotDataset instance
            train_indices: Indices to sample support set from
            test_indices: Indices to evaluate on
            k_shot: Number of support samples per class
            n_episodes: Number of support sets to sample and average
            
        Returns:
            Dictionary with averaged metrics
        """
        sampler = EpisodeSampler(n_way=2, k_shot=k_shot, n_query=1)
        
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for _ in range(n_episodes):
            # Sample support set
            support_set, _ = sampler.sample(dataset, train_indices)
            
            # Evaluate
            metrics = self.few_shot_evaluate(support_set, dataset, test_indices)
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        
        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def evaluate_multiple_k_shots(
        self,
        dataset: BotDataset,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        k_shots: List[int] = [1, 5, 10, 20],
        n_episodes: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate with multiple K-shot values.
        
        Args:
            dataset: BotDataset instance
            train_indices: Indices to sample support set from
            test_indices: Indices to evaluate on
            k_shots: List of K-shot values to evaluate
            n_episodes: Number of episodes per K-shot value
            
        Returns:
            Dictionary mapping K-shot value to metrics
        """
        results = {}
        
        for k in k_shots:
            metrics = self.evaluate_with_k_shot(
                dataset, train_indices, test_indices, k, n_episodes
            )
            results[k] = metrics
        
        return results
