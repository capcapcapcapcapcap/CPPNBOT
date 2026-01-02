"""
Evaluator: 少样本评估器

Implements Requirements 11.1, 11.2, 11.3, 11.4

Supports multi-modal evaluation with:
- Text data: online encoding or precomputed embeddings (faster)
- Graph data: precomputed embeddings (required for cross-domain evaluation)
- Few-shot evaluation with different K-shot values
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
    
    Multi-modal support:
    - Text data: online encoding or precomputed embeddings
    - Graph data: precomputed embeddings (在线编码需要完整图结构，跨域评估时不可用)
    """
    
    def __init__(
        self, 
        model: PrototypicalNetwork,
        enabled_modalities: Optional[List[str]] = None,
        use_precomputed_text_embeddings: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: PrototypicalNetwork instance (should be pre-trained)
            enabled_modalities: List of enabled modalities (default: ['num', 'cat'])
            use_precomputed_text_embeddings: Whether to use precomputed embeddings (default: True)
        """
        self.model = model
        
        # Multi-modal configuration
        self.enabled_modalities = enabled_modalities or ['num', 'cat']
        self.use_text = 'text' in self.enabled_modalities
        self.use_graph = 'graph' in self.enabled_modalities
        self.use_precomputed_text_embeddings = use_precomputed_text_embeddings
        
        # Device
        self.device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
        
        # Cache for text and graph data
        self._user_texts: Optional[Dict[int, str]] = None
        self._precomputed_text_embeddings: Optional[torch.Tensor] = None
        self._precomputed_graph_embeddings: Optional[torch.Tensor] = None
    
    def set_dataset_data(
        self,
        dataset: BotDataset
    ) -> None:
        """
        Load text and graph data from dataset.
        
        Args:
            dataset: BotDataset instance
        """
        # Load text data if text modality is enabled
        if self.use_text:
            if self.use_precomputed_text_embeddings and dataset.has_precomputed_text_embeddings():
                try:
                    self._precomputed_text_embeddings = dataset.get_text_embeddings().to(self.device)
                except FileNotFoundError:
                    self.use_precomputed_text_embeddings = False
            
            if not self.use_precomputed_text_embeddings:
                try:
                    self._user_texts = dataset.get_user_texts()
                except FileNotFoundError:
                    self._user_texts = None
                    self.use_text = False
        
        # Load precomputed graph embeddings if graph modality is enabled
        # 注意：图编码需要完整图结构，跨域评估时必须使用预计算嵌入
        if self.use_graph:
            if dataset.has_precomputed_graph_embeddings():
                try:
                    self._precomputed_graph_embeddings = dataset.get_graph_embeddings().to(self.device)
                except FileNotFoundError:
                    self._precomputed_graph_embeddings = None
                    self.use_graph = False
                    import logging
                    logging.warning(
                        f"Graph embeddings not found for {dataset.dataset_name}. "
                        f"Run: python precompute_graph_embeddings.py --dataset {dataset.dataset_name}"
                    )
            else:
                self._precomputed_graph_embeddings = None
                self.use_graph = False
                import logging
                logging.warning(
                    f"Graph modality enabled but no precomputed embeddings found. "
                    f"Disabling graph modality for evaluation."
                )
    
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
    
    def _get_graph_embeddings_for_indices(
        self, 
        indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get precomputed graph embeddings for given sample indices.
        
        Returns None if graph modality is disabled or embeddings not available.
        """
        if not self.use_graph or self._precomputed_graph_embeddings is None:
            return None
        
        return self._precomputed_graph_embeddings[indices]
    
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
        tp = ((predictions == positive_class) & (labels == positive_class)).sum().item()
        tn = ((predictions != positive_class) & (labels != positive_class)).sum().item()
        fp = ((predictions == positive_class) & (labels != positive_class)).sum().item()
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
        test_indices: torch.Tensor,
        support_indices: Optional[torch.Tensor] = None
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
            support_indices: Original indices of support samples (for text/graph)
            
        Returns:
            Dictionary with accuracy, precision, recall, and F1 score
        """
        self.model.eval()
        
        # Load dataset data if not already loaded
        if self.use_text and self._user_texts is None and self._precomputed_text_embeddings is None:
            self.set_dataset_data(test_dataset)
        if self.use_graph and self._precomputed_graph_embeddings is None:
            self.set_dataset_data(test_dataset)
        
        # Move support set to device
        support_set = self._move_to_device(support_set)
        
        # Get text data for support set (either precomputed embeddings or raw texts)
        support_texts = None
        support_text_embeddings = None
        if support_indices is not None:
            if self.use_precomputed_text_embeddings:
                support_text_embeddings = self._get_text_embeddings_for_indices(support_indices)
            else:
                support_texts = self._get_texts_for_indices(support_indices)
        
        # Collect all test samples
        test_num_features = []
        test_cat_features = []
        test_labels = []
        test_idx_list = []
        
        for idx in test_indices:
            item = test_dataset[idx.item()]
            test_num_features.append(item['num_features'])
            test_cat_features.append(item['cat_features'])
            test_labels.append(item['label'])
            test_idx_list.append(idx.item())
        
        # Stack into tensors
        query_set = {
            'num_features': torch.stack(test_num_features).to(self.device),
            'cat_features': torch.stack(test_cat_features).long().to(self.device)
        }
        test_labels = torch.stack(test_labels).to(self.device)
        
        # Get text data for query set (either precomputed embeddings or raw texts)
        query_texts = None
        query_text_embeddings = None
        if self.use_text:
            test_idx_tensor = torch.tensor(test_idx_list)
            if self.use_precomputed_text_embeddings:
                query_text_embeddings = self._get_text_embeddings_for_indices(test_idx_tensor)
            else:
                query_texts = self._get_texts_for_indices(test_idx_tensor)
        
        # Get graph embeddings (precomputed)
        support_graph_embeddings = None
        query_graph_embeddings = None
        if self.use_graph and support_indices is not None:
            support_graph_embeddings = self._get_graph_embeddings_for_indices(support_indices)
        if self.use_graph:
            test_idx_tensor = torch.tensor(test_idx_list)
            query_graph_embeddings = self._get_graph_embeddings_for_indices(test_idx_tensor)
        
        # Forward pass with frozen encoder
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
        # Load dataset data
        self.set_dataset_data(dataset)
        
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
            
            # Get support indices from sampler
            support_indices = getattr(sampler, '_last_support_indices', None)
            
            # Evaluate
            metrics = self.few_shot_evaluate(
                support_set, dataset, test_indices, support_indices
            )
            
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
