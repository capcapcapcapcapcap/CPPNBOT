"""
MetaTrainer: 元训练器

Implements Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6

Supports multi-modal training with:
- Text data loading and passing
- Graph data loading and passing
- Selective text backbone freezing
- Separate learning rates for text encoder
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    
    Multi-modal support:
    - Text data loading and encoding
    - Graph data loading and encoding
    - Selective text backbone freezing
    - Separate learning rates for text encoder
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
            config: Configuration dictionary
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
        
        # Multi-modal configuration
        self.enabled_modalities = config.get('enabled_modalities', ['num', 'cat'])
        self.use_text = 'text' in self.enabled_modalities
        self.use_graph = 'graph' in self.enabled_modalities

        # Load text data if text modality is enabled
        self._user_texts: Optional[Dict[int, str]] = None
        if self.use_text:
            try:
                self._user_texts = dataset.get_user_texts()
                logger.info(f"Loaded {len(self._user_texts)} user texts for text encoding")
            except FileNotFoundError:
                logger.warning("User texts not found, disabling text modality")
                self.use_text = False
                self.enabled_modalities = [m for m in self.enabled_modalities if m != 'text']
        
        # Graph data (loaded from dataset)
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_type: Optional[torch.Tensor] = None
        if self.use_graph:
            self._edge_index = dataset.edge_index
            self._edge_type = dataset.edge_type
            logger.info(f"Loaded graph with {self._edge_index.size(1)} edges for graph encoding")
        
        # Device
        self.device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
        
        # Move graph data to device
        if self._edge_index is not None:
            self._edge_index = self._edge_index.to(self.device)
        if self._edge_type is not None:
            self._edge_type = self._edge_type.to(self.device)
        
        # Setup optimizer with separate learning rates for text encoder
        self.optimizer = self._setup_optimizer(config)
        
        # Loss function (negative log-likelihood)
        self.criterion = nn.NLLLoss()
        
        # Get split indices
        self.train_indices = dataset.get_split_indices('train')
        self.val_indices = dataset.get_split_indices('val')
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        
        # Handle text backbone freezing
        text_freeze_backbone = config.get('text_freeze_backbone', True)
        if self.use_text and hasattr(model.encoder, 'freeze_text_backbone'):
            if text_freeze_backbone:
                model.encoder.freeze_text_backbone()
                logger.info("Text encoder backbone frozen")
            else:
                model.encoder.unfreeze_text_backbone()
                logger.info("Text encoder backbone unfrozen")

    def _setup_optimizer(self, config: Dict) -> optim.Optimizer:
        """Setup optimizer with separate learning rates for text encoder."""
        learning_rate = config.get('learning_rate', 1e-3)
        text_learning_rate = config.get('text_learning_rate', 1e-5)
        weight_decay = config.get('weight_decay', 1e-4)
        
        # Check if model has text encoder with separate parameters
        if self.use_text and hasattr(self.model.encoder, 'text_encoder') and self.model.encoder.text_encoder is not None:
            # Separate parameters for text encoder
            text_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if 'text_encoder' in name:
                    text_params.append(param)
                else:
                    other_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': other_params, 'lr': learning_rate},
                {'params': text_params, 'lr': text_learning_rate}
            ]
            
            logger.info(f"Using separate learning rates: main={learning_rate}, text={text_learning_rate}")
            return optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            # Standard optimizer for all parameters
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

    def _move_to_device(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move data dictionary to the model's device."""
        return {k: v.to(self.device) for k, v in data.items()}
    
    def _get_texts_for_indices(self, indices: torch.Tensor) -> Optional[List[str]]:
        """Get text data for given sample indices.
        
        Handles user_texts.json format which contains dictionaries with
        'description' and 'tweets' fields.
        """
        if not self.use_text or self._user_texts is None:
            return None
        
        texts = []
        for idx in indices.tolist():
            text_data = self._user_texts.get(idx, "")
            
            # Handle dictionary format from user_texts.json
            if isinstance(text_data, dict):
                # Combine description and tweets into a single string
                description = text_data.get('description', '') or ''
                tweets = text_data.get('tweets', []) or []
                
                # Join tweets with space
                tweets_text = ' '.join(tweets) if tweets else ''
                
                # Combine description and tweets
                combined = f"{description} {tweets_text}".strip()
                texts.append(combined if combined else "")
            elif isinstance(text_data, str):
                texts.append(text_data if text_data else "")
            else:
                texts.append("")
        
        return texts
    
    def _get_subgraph_for_indices(
        self, 
        indices: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get subgraph data for given sample indices."""
        if not self.use_graph or self._edge_index is None:
            return None, None
        
        # Return full graph - the model will select relevant nodes
        return self._edge_index, self._edge_type

    def _compute_episode_loss(
        self,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor],
        support_indices: Optional[torch.Tensor] = None,
        query_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss for a single episode."""
        # Move data to device
        support_set = self._move_to_device(support_set)
        query_set = self._move_to_device(query_set)
        
        # Get text data if enabled
        support_texts = None
        query_texts = None
        if support_indices is not None:
            support_texts = self._get_texts_for_indices(support_indices)
        if query_indices is not None:
            query_texts = self._get_texts_for_indices(query_indices)
        
        # Get graph data if enabled
        edge_index, edge_type = None, None
        if support_indices is not None and query_indices is not None:
            edge_index, edge_type = self._get_subgraph_for_indices(
                torch.cat([support_indices, query_indices])
            )
        
        # Forward pass with multi-modal data
        output = self.model(
            support_set, 
            query_set,
            support_texts=support_texts,
            query_texts=query_texts,
            edge_index=edge_index,
            edge_type=edge_type
        )
        log_probs = output['log_probs']
        
        # Compute NLL loss
        query_labels = query_set['labels']
        loss = self.criterion(log_probs, query_labels)
        
        return loss
    
    def _compute_episode_metrics(
        self,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor],
        support_indices: Optional[torch.Tensor] = None,
        query_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute metrics for a single episode."""
        # Move data to device
        support_set = self._move_to_device(support_set)
        query_set = self._move_to_device(query_set)
        
        # Get text data if enabled
        support_texts = None
        query_texts = None
        if support_indices is not None:
            support_texts = self._get_texts_for_indices(support_indices)
        if query_indices is not None:
            query_texts = self._get_texts_for_indices(query_indices)
        
        # Get graph data if enabled
        edge_index, edge_type = None, None
        if support_indices is not None and query_indices is not None:
            edge_index, edge_type = self._get_subgraph_for_indices(
                torch.cat([support_indices, query_indices])
            )
        
        # Forward pass
        with torch.no_grad():
            output = self.model(
                support_set, 
                query_set,
                support_texts=support_texts,
                query_texts=query_texts,
                edge_index=edge_index,
                edge_type=edge_type
            )
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
        """Train for one epoch by sampling multiple episodes."""
        self.model.train()
        
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        for _ in range(n_episodes):
            # Sample episode (with indices for multi-modal data)
            support_set, query_set = self.sampler.sample(
                self.dataset, self.train_indices
            )
            
            # Get original indices from sampler if available
            support_indices = getattr(self.sampler, '_last_support_indices', None)
            query_indices = getattr(self.sampler, '_last_query_indices', None)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self._compute_episode_loss(
                support_set, query_set, 
                support_indices, query_indices
            )
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Compute metrics (without gradients)
            metrics = self._compute_episode_metrics(
                support_set, query_set,
                support_indices, query_indices
            )
            for k in total_metrics:
                total_metrics[k] += metrics[k]
        
        result = {'loss': total_loss / n_episodes}
        for k in total_metrics:
            result[k] = total_metrics[k] / n_episodes
        
        return result
    
    def validate(self, n_episodes: int) -> Dict[str, float]:
        """Validate on validation set episodes."""
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        with torch.no_grad():
            for _ in range(n_episodes):
                # Sample episode from validation set
                support_set, query_set = self.sampler.sample(
                    self.dataset, self.val_indices
                )
                
                # Get original indices from sampler if available
                support_indices = getattr(self.sampler, '_last_support_indices', None)
                query_indices = getattr(self.sampler, '_last_query_indices', None)
                
                # Move data to device
                support_set = self._move_to_device(support_set)
                query_set = self._move_to_device(query_set)
                
                # Get text data if enabled
                support_texts = None
                query_texts = None
                if support_indices is not None:
                    support_texts = self._get_texts_for_indices(support_indices)
                if query_indices is not None:
                    query_texts = self._get_texts_for_indices(query_indices)
                
                # Get graph data if enabled
                edge_index, edge_type = None, None
                if support_indices is not None and query_indices is not None:
                    edge_index, edge_type = self._get_subgraph_for_indices(
                        torch.cat([support_indices, query_indices])
                    )
                
                # Forward pass
                output = self.model(
                    support_set, 
                    query_set,
                    support_texts=support_texts,
                    query_texts=query_texts,
                    edge_index=edge_index,
                    edge_type=edge_type
                )
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
        """Save best model checkpoint only."""
        if not is_best:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'enabled_modalities': self.enabled_modalities
        }
        
        best_path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load enabled modalities if available
        if 'enabled_modalities' in checkpoint:
            saved_modalities = checkpoint['enabled_modalities']
            if saved_modalities != self.enabled_modalities:
                logger.warning(
                    f"Checkpoint modalities {saved_modalities} differ from current {self.enabled_modalities}"
                )
        
        logger.info(f"Loaded checkpoint from {filepath} (epoch {self.current_epoch})")
    
    def train(self, n_epochs: Optional[int] = None) -> Dict[str, list]:
        """Complete training flow with validation and early stopping."""
        if n_epochs is None:
            n_epochs = self.n_epochs
        
        history = {
            'train_loss': [], 'train_accuracy': [], 'train_precision': [], 
            'train_recall': [], 'train_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': []
        }
        
        # Log training configuration
        modality_str = ', '.join(self.enabled_modalities)
        logger.info(f"Training: {n_epochs} epochs, {self.n_episodes_train} episodes/epoch")
        logger.info(f"Enabled modalities: {modality_str}")
        
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
            
            # Log progress
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
