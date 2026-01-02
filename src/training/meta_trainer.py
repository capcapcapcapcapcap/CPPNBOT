"""
MetaTrainer: 元训练器

Implements Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6

Supports multi-modal training with:
- Text data loading and passing (online encoding or precomputed embeddings)
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
        
        # Text embedding mode: precomputed (fast) or online encoding
        self.use_precomputed_text_embeddings = config.get('use_precomputed_text_embeddings', True)
        self._precomputed_text_embeddings: Optional[torch.Tensor] = None

        # Load text data if text modality is enabled
        self._user_texts: Optional[Dict[int, str]] = None
        if self.use_text:
            if self.use_precomputed_text_embeddings and dataset.has_precomputed_text_embeddings():
                # 使用预计算的文本嵌入 (推荐，更快)
                try:
                    self._precomputed_text_embeddings = dataset.get_text_embeddings()
                    logger.info(f"Loaded precomputed text embeddings: {self._precomputed_text_embeddings.shape}")
                except FileNotFoundError as e:
                    logger.warning(f"Precomputed embeddings not found: {e}")
                    logger.warning("Falling back to online text encoding (slower)")
                    self.use_precomputed_text_embeddings = False
            else:
                self.use_precomputed_text_embeddings = False
            
            # 如果没有预计算嵌入，加载原始文本用于在线编码
            if not self.use_precomputed_text_embeddings:
                try:
                    self._user_texts = dataset.get_user_texts()
                    logger.info(f"Loaded {len(self._user_texts)} user texts for online encoding")
                except FileNotFoundError:
                    logger.warning("User texts not found, disabling text modality")
                    self.use_text = False
                    self.enabled_modalities = [m for m in self.enabled_modalities if m != 'text']
        
        # Graph data (loaded from dataset)
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_type: Optional[torch.Tensor] = None
        self._precomputed_graph_embeddings: Optional[torch.Tensor] = None
        
        if self.use_graph:
            # 优先使用预计算的图嵌入
            if dataset.has_precomputed_graph_embeddings():
                try:
                    self._precomputed_graph_embeddings = dataset.get_graph_embeddings()
                    logger.info(f"Loaded precomputed graph embeddings: {self._precomputed_graph_embeddings.shape}")
                except FileNotFoundError as e:
                    logger.warning(f"Precomputed graph embeddings not found: {e}")
                    logger.warning("Will compute graph embeddings at initialization (slower)")
            
            # 如果没有预计算嵌入，加载图数据用于在线计算
            if self._precomputed_graph_embeddings is None:
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
        
        # Move precomputed embeddings to device
        if self._precomputed_text_embeddings is not None:
            self._precomputed_text_embeddings = self._precomputed_text_embeddings.to(self.device)
        if self._precomputed_graph_embeddings is not None:
            self._precomputed_graph_embeddings = self._precomputed_graph_embeddings.to(self.device)
        
        # Precompute graph embeddings if graph modality is enabled but no precomputed file
        # 如果没有预计算的图嵌入文件，在初始化时计算
        if self.use_graph and self._precomputed_graph_embeddings is None and self._edge_index is not None:
            self._precompute_graph_embeddings(dataset)
        
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
        
        # Handle text backbone freezing (only for online encoding mode)
        if self.use_text and not self.use_precomputed_text_embeddings:
            text_freeze_backbone = config.get('text_freeze_backbone', True)
            if hasattr(model.encoder, 'freeze_text_backbone'):
                if text_freeze_backbone:
                    model.encoder.freeze_text_backbone()
                    logger.info("Text encoder backbone frozen")
                else:
                    model.encoder.unfreeze_text_backbone()
                    logger.info("Text encoder backbone unfrozen")

    def _precompute_graph_embeddings(self, dataset: BotDataset) -> None:
        """预计算所有节点的图嵌入
        
        图编码器输入固定为 num + cat 特征，与文本模态独立。
        在训练前预计算所有节点的图嵌入，然后在 episode 中只选择相关节点的嵌入。
        """
        logger.info("Precomputing graph embeddings for all nodes...")
        
        # 构建所有节点的输入特征（num + cat）
        with torch.no_grad():
            # 数值特征
            num_features = dataset.num_features.to(self.device)
            num_embed = self.model.encoder.numerical_encoder(num_features) if self.model.encoder.numerical_encoder else None
            
            # 分类特征
            cat_features = dataset.cat_features.to(self.device)
            cat_embed = self.model.encoder.categorical_encoder(cat_features) if self.model.encoder.categorical_encoder else None
            
            # 拼接 num + cat 作为图编码器输入
            available_embeds = [e for e in [num_embed, cat_embed] if e is not None]
            if available_embeds:
                graph_input = torch.cat(available_embeds, dim=1)
            else:
                num_users = len(dataset)
                graph_input = torch.zeros(num_users, self.model.encoder.graph_encoder.input_dim, device=self.device)
            
            # 图编码
            self._precomputed_graph_embeddings = self.model.encoder.graph_encoder(
                graph_input, self._edge_index, self._edge_type
            )
        
        logger.info(f"Precomputed graph embeddings: {self._precomputed_graph_embeddings.shape}")

    def _setup_optimizer(self, config: Dict) -> optim.Optimizer:
        """Setup optimizer with separate learning rates for text encoder."""
        learning_rate = config.get('learning_rate', 1e-3)
        text_learning_rate = config.get('text_learning_rate', 1e-5)
        weight_decay = config.get('weight_decay', 1e-4)
        
        # 保存学习率用于调度器
        self.base_lr = learning_rate
        
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
        """Get text data for given sample indices (for online encoding mode).
        
        Handles user_texts.json format which contains dictionaries with
        'description' and 'tweets' fields.
        
        Returns None if using precomputed embeddings or text modality is disabled.
        """
        if not self.use_text or self.use_precomputed_text_embeddings or self._user_texts is None:
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
        """Get subgraph data for given sample indices.
        
        注意：当使用预计算图嵌入时，此方法不再使用。
        """
        if not self.use_graph or self._edge_index is None:
            return None, None
        
        # Return full graph - the model will select relevant nodes
        return self._edge_index, self._edge_type
    
    def _get_graph_embeddings_for_indices(
        self, 
        indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get precomputed graph embeddings for given sample indices.
        
        Returns None if graph modality is disabled or embeddings not precomputed.
        """
        if not self.use_graph or self._precomputed_graph_embeddings is None:
            return None
        
        return self._precomputed_graph_embeddings[indices]
    
    def _get_text_embeddings_for_indices(
        self, 
        indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get precomputed text embeddings for given sample indices.
        
        Returns None if not using precomputed embeddings or text modality is disabled.
        """
        if not self.use_text or not self.use_precomputed_text_embeddings:
            return None
        if self._precomputed_text_embeddings is None:
            return None
        
        return self._precomputed_text_embeddings[indices]

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
        
        # Get text data if enabled (either precomputed embeddings or raw texts)
        support_texts = None
        query_texts = None
        support_text_embeddings = None
        query_text_embeddings = None
        
        if support_indices is not None:
            if self.use_precomputed_text_embeddings:
                support_text_embeddings = self._get_text_embeddings_for_indices(support_indices)
            else:
                support_texts = self._get_texts_for_indices(support_indices)
        if query_indices is not None:
            if self.use_precomputed_text_embeddings:
                query_text_embeddings = self._get_text_embeddings_for_indices(query_indices)
            else:
                query_texts = self._get_texts_for_indices(query_indices)
        
        # Get graph embeddings if enabled (precomputed)
        support_graph_embeddings = None
        query_graph_embeddings = None
        if self.use_graph and support_indices is not None:
            support_graph_embeddings = self._get_graph_embeddings_for_indices(support_indices)
        if self.use_graph and query_indices is not None:
            query_graph_embeddings = self._get_graph_embeddings_for_indices(query_indices)
        
        # Forward pass with multi-modal data
        output = self.model(
            support_set, 
            query_set,
            support_texts=support_texts,
            query_texts=query_texts,
            support_text_embeddings=support_text_embeddings,
            query_text_embeddings=query_text_embeddings,
            support_graph_embeddings=support_graph_embeddings,
            query_graph_embeddings=query_graph_embeddings,
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
        
        # Get text data if enabled (either precomputed embeddings or raw texts)
        support_texts = None
        query_texts = None
        support_text_embeddings = None
        query_text_embeddings = None
        
        if support_indices is not None:
            if self.use_precomputed_text_embeddings:
                support_text_embeddings = self._get_text_embeddings_for_indices(support_indices)
            else:
                support_texts = self._get_texts_for_indices(support_indices)
        if query_indices is not None:
            if self.use_precomputed_text_embeddings:
                query_text_embeddings = self._get_text_embeddings_for_indices(query_indices)
            else:
                query_texts = self._get_texts_for_indices(query_indices)
        
        # Get graph embeddings if enabled (precomputed)
        support_graph_embeddings = None
        query_graph_embeddings = None
        if self.use_graph and support_indices is not None:
            support_graph_embeddings = self._get_graph_embeddings_for_indices(support_indices)
        if self.use_graph and query_indices is not None:
            query_graph_embeddings = self._get_graph_embeddings_for_indices(query_indices)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(
                support_set, 
                query_set,
                support_texts=support_texts,
                query_texts=query_texts,
                support_text_embeddings=support_text_embeddings,
                query_text_embeddings=query_text_embeddings,
                support_graph_embeddings=support_graph_embeddings,
                query_graph_embeddings=query_graph_embeddings,
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
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
                
                # Get text data if enabled (either precomputed embeddings or raw texts)
                support_texts = None
                query_texts = None
                support_text_embeddings = None
                query_text_embeddings = None
                
                if support_indices is not None:
                    if self.use_precomputed_text_embeddings:
                        support_text_embeddings = self._get_text_embeddings_for_indices(support_indices)
                    else:
                        support_texts = self._get_texts_for_indices(support_indices)
                if query_indices is not None:
                    if self.use_precomputed_text_embeddings:
                        query_text_embeddings = self._get_text_embeddings_for_indices(query_indices)
                    else:
                        query_texts = self._get_texts_for_indices(query_indices)
                
                # Get graph embeddings if enabled (precomputed)
                support_graph_embeddings = None
                query_graph_embeddings = None
                if self.use_graph and support_indices is not None:
                    support_graph_embeddings = self._get_graph_embeddings_for_indices(support_indices)
                if self.use_graph and query_indices is not None:
                    query_graph_embeddings = self._get_graph_embeddings_for_indices(query_indices)
                
                # Forward pass
                output = self.model(
                    support_set, 
                    query_set,
                    support_texts=support_texts,
                    query_texts=query_texts,
                    support_text_embeddings=support_text_embeddings,
                    query_text_embeddings=query_text_embeddings,
                    support_graph_embeddings=support_graph_embeddings,
                    query_graph_embeddings=query_graph_embeddings,
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
        
        # 学习率调度器: ReduceLROnPlateau - 当验证loss停滞时降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,      # 每次降低50%
            patience=5,      # 5个epoch无改善则降低
            min_lr=1e-6,
            verbose=False
        )
        
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
            
            # 更新学习率调度器
            scheduler.step(val_metrics['loss'])
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log progress
            improved = val_metrics['loss'] < self.best_val_loss
            marker = " *" if improved else ""
            lr_info = f" lr={current_lr:.1e}" if current_lr < self.base_lr else ""
            logger.info(
                f"Epoch {self.current_epoch:3d}/{n_epochs} | "
                f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
                f"F1: {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f}{marker}{lr_info}"
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
