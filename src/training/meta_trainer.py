"""
MetaTrainer: 元训练器

Implements Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..data.dataset import BotDataset
from ..data.episode_sampler import EpisodeSampler
from ..models.prototypical import PrototypicalNetwork


logger = logging.getLogger(__name__)


class MetaTrainer:
    """元训练器
    
    Implements episodic meta-learning training for prototypical networks.
    Supports training with multiple episodes per epoch, validation,
    checkpoint saving, and early stopping.
    """
    
    def __init__(
        self,
        model: PrototypicalNetwork,
        dataset: BotDataset,
        config: Dict
    ):
        """
        Initialize the meta trainer.
        
        Args:
            model: PrototypicalNetwork instance
            dataset: BotDataset for training
            config: Configuration dictionary with keys:
                - n_way: Number of classes per episode
                - k_shot: Number of support samples per class
                - n_query: Number of query samples per class
                - n_episodes_train: Episodes per training epoch
                - n_episodes_val: Episodes for validation
                - n_epochs: Total training epochs
                - learning_rate: Optimizer learning rate
                - weight_decay: Optimizer weight decay
                - patience: Early stopping patience
                - output_dir: Directory for saving checkpoints
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Episode sampler
        self.sampler = EpisodeSampler(
            n_way=config.get('n_way', 2),
            k_shot=config.get('k_shot', 5),
            n_query=config.get('n_query', 15)
        )
        
        # Training parameters
        self.n_episodes_train = config.get('n_episodes_train', 100)
        self.n_episodes_val = config.get('n_episodes_val', 50)
        self.n_epochs = config.get('n_epochs', 100)
        self.patience = config.get('patience', 10)
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Loss function (negative log-likelihood)
        self.criterion = nn.NLLLoss()
        
        # Get split indices
        self.train_indices = dataset.get_split_indices('train')
        self.val_indices = dataset.get_split_indices('val')
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        
        # Device
        self.device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')

    def _move_to_device(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move data dictionary to the model's device."""
        return {k: v.to(self.device) for k, v in data.items()}
    
    def _compute_episode_loss(
        self,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss for a single episode.
        
        Args:
            support_set: Support set with features and labels
            query_set: Query set with features and labels
            
        Returns:
            Scalar loss tensor (negative log-likelihood on query set)
        """
        # Move data to device
        support_set = self._move_to_device(support_set)
        query_set = self._move_to_device(query_set)
        
        # Forward pass
        output = self.model(support_set, query_set)
        log_probs = output['log_probs']
        
        # Compute NLL loss
        query_labels = query_set['labels']
        loss = self.criterion(log_probs, query_labels)
        
        return loss
    
    def _compute_episode_metrics(
        self,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute metrics for a single episode.
        
        Args:
            support_set: Support set with features and labels
            query_set: Query set with features and labels
            
        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        # Move data to device
        support_set = self._move_to_device(support_set)
        query_set = self._move_to_device(query_set)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(support_set, query_set)
            log_probs = output['log_probs']
            predictions = log_probs.argmax(dim=1)
            
            query_labels = query_set['labels']
            
            # Compute confusion matrix components (for bot class = 1)
            tp = ((predictions == 1) & (query_labels == 1)).sum().item()
            fp = ((predictions == 1) & (query_labels == 0)).sum().item()
            fn = ((predictions == 0) & (query_labels == 1)).sum().item()
            tn = ((predictions == 0) & (query_labels == 0)).sum().item()
            
            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_epoch(self, n_episodes: int) -> Dict[str, float]:
        """
        Train for one epoch by sampling multiple episodes.
        
        Args:
            n_episodes: Number of episodes to sample and train on
            
        Returns:
            Dictionary with loss, accuracy, precision, recall, f1
        """
        self.model.train()
        
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        for _ in range(n_episodes):
            # Sample episode
            support_set, query_set = self.sampler.sample(
                self.dataset, self.train_indices
            )
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self._compute_episode_loss(support_set, query_set)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Compute metrics (without gradients)
            metrics = self._compute_episode_metrics(support_set, query_set)
            for k in total_metrics:
                total_metrics[k] += metrics[k]
        
        result = {'loss': total_loss / n_episodes}
        for k in total_metrics:
            result[k] = total_metrics[k] / n_episodes
        
        return result
    
    def validate(self, n_episodes: int) -> Dict[str, float]:
        """
        Validate on validation set episodes.
        
        Args:
            n_episodes: Number of episodes to sample for validation
            
        Returns:
            Dictionary with loss, accuracy, precision, recall, f1
        """
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        with torch.no_grad():
            for _ in range(n_episodes):
                # Sample episode from validation set
                support_set, query_set = self.sampler.sample(
                    self.dataset, self.val_indices
                )
                
                # Move data to device
                support_set = self._move_to_device(support_set)
                query_set = self._move_to_device(query_set)
                
                # Forward pass
                output = self.model(support_set, query_set)
                log_probs = output['log_probs']
                
                # Compute loss
                query_labels = query_set['labels']
                loss = self.criterion(log_probs, query_labels)
                total_loss += loss.item()
                
                # Compute metrics
                predictions = log_probs.argmax(dim=1)
                tp = ((predictions == 1) & (query_labels == 1)).sum().item()
                fp = ((predictions == 1) & (query_labels == 0)).sum().item()
                fn = ((predictions == 0) & (query_labels == 1)).sum().item()
                tn = ((predictions == 0) & (query_labels == 0)).sum().item()
                
                total = tp + fp + fn + tn
                total_metrics['accuracy'] += (tp + tn) / total if total > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                total_metrics['precision'] += precision
                total_metrics['recall'] += recall
                total_metrics['f1'] += 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result = {'loss': total_loss / n_episodes}
        for k in total_metrics:
            result[k] = total_metrics[k] / n_episodes
        
        return result

    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save best model checkpoint only.
        
        Args:
            is_best: If True, saves as 'best_model.pt'
        """
        if not is_best:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        best_path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {filepath} (epoch {self.current_epoch})")
    
    def train(self, n_epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Complete training flow with validation and early stopping.
        
        Args:
            n_epochs: Number of epochs to train. If None, uses config value.
            
        Returns:
            Dictionary with training history:
                - 'train_loss': List of training losses per epoch
                - 'train_accuracy': List of training accuracies per epoch
                - 'val_loss': List of validation losses per epoch
                - 'val_accuracy': List of validation accuracies per epoch
        """
        if n_epochs is None:
            n_epochs = self.n_epochs
        
        history = {
            'train_loss': [], 'train_accuracy': [], 'train_precision': [], 
            'train_recall': [], 'train_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': []
        }
        
        logger.info(f"Training: {n_epochs} epochs, {self.n_episodes_train} episodes/epoch")
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch + 1
            
            # Training
            train_metrics = self.train_epoch(self.n_episodes_train)
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
            
            # Validation
            val_metrics = self.validate(self.n_episodes_val)
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)
            
            # Log progress (简洁格式)
            improved = val_metrics['loss'] < self.best_val_loss
            marker = " *" if improved else ""
            logger.info(
                f"Epoch {self.current_epoch:3d}/{n_epochs} | "
                f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
                f"F1: {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f}{marker}"
            )
            
            # Check for improvement
            if improved:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {self.current_epoch} (no improvement for {self.patience} epochs)")
                break
        
        logger.info(f"Done. Best val loss: {self.best_val_loss:.4f}")
        return history
