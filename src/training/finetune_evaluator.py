"""
FinetuneEvaluator: 带微调的少样本评估器

在目标域评估时，先用 support set 微调模型几步，
然后再进行预测。这比原型网络的"只换原型"适应能力更强。

两种微调策略:
1. 全模型微调: 更新所有参数
2. 只微调分类头: 冻结编码器，只更新一个新的分类层
"""

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..data.dataset import BotDataset
from ..data.episode_sampler import EpisodeSampler
from ..models.prototypical import PrototypicalNetwork


logger = logging.getLogger(__name__)


class FinetuneEvaluator:
    """带微调的少样本评估器"""
    
    def __init__(
        self,
        model: PrototypicalNetwork,
        enabled_modalities: Optional[List[str]] = None,
        use_precomputed_text_embeddings: bool = True,
        finetune_steps: int = 10,
        finetune_lr: float = 1e-4,
        finetune_mode: str = 'full'  # 'full' or 'head'
    ):
        """
        Args:
            model: 预训练的 PrototypicalNetwork
            enabled_modalities: 启用的模态
            use_precomputed_text_embeddings: 是否使用预计算文本嵌入
            finetune_steps: 微调步数
            finetune_lr: 微调学习率
            finetune_mode: 'full' 全模型微调, 'head' 只微调分类头
        """
        self.model = model
        self.enabled_modalities = enabled_modalities or ['num', 'cat']
        self.use_text = 'text' in self.enabled_modalities
        self.use_graph = 'graph' in self.enabled_modalities
        self.use_precomputed_text_embeddings = use_precomputed_text_embeddings
        
        self.finetune_steps = finetune_steps
        self.finetune_lr = finetune_lr
        self.finetune_mode = finetune_mode
        
        self.device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
        
        # 缓存
        self._precomputed_text_embeddings: Optional[torch.Tensor] = None
        self._precomputed_graph_embeddings: Optional[torch.Tensor] = None
    
    def set_dataset_data(self, dataset: BotDataset) -> None:
        """加载数据集的预计算嵌入"""
        if self.use_text and dataset.has_precomputed_text_embeddings():
            try:
                self._precomputed_text_embeddings = dataset.get_text_embeddings().to(self.device)
            except FileNotFoundError:
                self.use_text = False
        
        if self.use_graph and dataset.has_precomputed_graph_embeddings():
            try:
                self._precomputed_graph_embeddings = dataset.get_graph_embeddings().to(self.device)
            except FileNotFoundError:
                self.use_graph = False
    
    def _get_text_embeddings(self, indices: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_text or self._precomputed_text_embeddings is None:
            return None
        return self._precomputed_text_embeddings[indices]
    
    def _get_graph_embeddings(self, indices: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_graph or self._precomputed_graph_embeddings is None:
            return None
        return self._precomputed_graph_embeddings[indices]
    
    def _create_finetune_model(self) -> nn.Module:
        """创建用于微调的模型副本"""
        if self.finetune_mode == 'full':
            # 全模型微调：复制整个模型
            return copy.deepcopy(self.model)
        else:
            # 只微调分类头：冻结编码器，添加新的分类层
            return FinetuneHead(self.model, self.model.encoder.output_dim)
    
    def _finetune_on_support(
        self,
        model: nn.Module,
        support_set: Dict[str, torch.Tensor],
        support_indices: torch.Tensor
    ) -> nn.Module:
        """在 support set 上微调模型"""
        model.train()
        
        # 获取多模态数据
        support_text_emb = self._get_text_embeddings(support_indices)
        support_graph_emb = self._get_graph_embeddings(support_indices)
        
        # 设置优化器
        if self.finetune_mode == 'full':
            optimizer = optim.Adam(model.parameters(), lr=self.finetune_lr)
        else:
            # 只优化分类头
            optimizer = optim.Adam(model.classifier.parameters(), lr=self.finetune_lr)
        
        criterion = nn.CrossEntropyLoss()
        labels = support_set['labels']
        
        for step in range(self.finetune_steps):
            optimizer.zero_grad()
            
            if self.finetune_mode == 'full':
                # 全模型：用编码器提取特征，然后用原型方式或直接分类
                # 这里我们用一个简单的方式：把 support 当作 query 自己预测自己
                # 用交叉熵损失而不是原型距离
                features = model.encoder(
                    support_set,
                    text_embeddings=support_text_emb,
                    graph_embeddings=support_graph_emb
                )
                # 添加一个临时分类头
                if not hasattr(model, '_temp_classifier'):
                    model._temp_classifier = nn.Linear(features.size(1), 2).to(self.device)
                logits = model._temp_classifier(features)
            else:
                # 分类头模式
                logits = model(
                    support_set,
                    support_text_emb,
                    support_graph_emb
                )
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        return model
    
    def _predict_with_finetuned(
        self,
        model: nn.Module,
        query_set: Dict[str, torch.Tensor],
        query_indices: torch.Tensor
    ) -> torch.Tensor:
        """用微调后的模型预测"""
        query_text_emb = self._get_text_embeddings(query_indices)
        query_graph_emb = self._get_graph_embeddings(query_indices)
        
        with torch.no_grad():
            if self.finetune_mode == 'full':
                features = model.encoder(
                    query_set,
                    text_embeddings=query_text_emb,
                    graph_embeddings=query_graph_emb
                )
                logits = model._temp_classifier(features)
            else:
                logits = model(
                    query_set,
                    query_text_emb,
                    query_graph_emb
                )
            
            predictions = logits.argmax(dim=1)
        
        return predictions
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """计算评估指标"""
        predictions = predictions.cpu()
        labels = labels.cpu()
        
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        tn = ((predictions == 0) & (labels == 0)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()
        
        total = tp + tn + fp + fn
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
    
    def evaluate_with_finetune(
        self,
        dataset: BotDataset,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        k_shot: int,
        n_episodes: int = 100
    ) -> Dict[str, float]:
        """带微调的评估"""
        self.set_dataset_data(dataset)
        
        sampler = EpisodeSampler(n_way=2, k_shot=k_shot, n_query=1)
        
        all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for ep in range(n_episodes):
            # 采样 support set
            support_set, _ = sampler.sample(dataset, train_indices)
            support_indices = sampler._last_support_indices
            
            # 移动到设备
            support_set = {k: v.to(self.device) for k, v in support_set.items()}
            
            # 创建模型副本并微调
            ft_model = self._create_finetune_model()
            ft_model = self._finetune_on_support(ft_model, support_set, support_indices)
            
            # 构建测试集
            test_num = []
            test_cat = []
            test_labels = []
            test_idx_list = []
            
            for idx in test_indices:
                item = dataset[idx.item()]
                test_num.append(item['num_features'])
                test_cat.append(item['cat_features'])
                test_labels.append(item['label'])
                test_idx_list.append(idx.item())
            
            query_set = {
                'num_features': torch.stack(test_num).to(self.device),
                'cat_features': torch.stack(test_cat).long().to(self.device)
            }
            test_labels = torch.stack(test_labels).to(self.device)
            query_indices = torch.tensor(test_idx_list)
            
            # 预测
            predictions = self._predict_with_finetuned(ft_model, query_set, query_indices)
            
            # 计算指标
            metrics = self._compute_metrics(predictions, test_labels)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # 清理
            del ft_model
        
        # 平均
        return {k: sum(v) / len(v) for k, v in all_metrics.items()}
    
    def evaluate_multiple_k_shots(
        self,
        dataset: BotDataset,
        train_indices: torch.Tensor,
        test_indices: torch.Tensor,
        k_shots: List[int] = [1, 5, 10, 20],
        n_episodes: int = 100
    ) -> Dict[int, Dict[str, float]]:
        """评估多个 k-shot 值"""
        results = {}
        for k in k_shots:
            logger.info(f"Evaluating {k}-shot with {self.finetune_steps} finetune steps...")
            metrics = self.evaluate_with_finetune(
                dataset, train_indices, test_indices, k, n_episodes
            )
            results[k] = metrics
            logger.info(f"{k:2d}-shot | Acc: {metrics['accuracy']:.4f} | "
                       f"F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
        return results


class FinetuneHead(nn.Module):
    """只微调分类头的包装器"""
    
    def __init__(self, base_model: PrototypicalNetwork, feature_dim: int):
        super().__init__()
        self.encoder = base_model.encoder
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        text_embeddings: Optional[torch.Tensor] = None,
        graph_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(
                batch,
                text_embeddings=text_embeddings,
                graph_embeddings=graph_embeddings
            )
        return self.classifier(features)
