"""
EpisodeSampler: N-way K-shot Episode 采样器

Implements Requirements 2.1, 2.2, 2.3, 2.4
"""

from typing import Dict, Tuple

import torch

from .dataset import BotDataset


class InsufficientSamplesError(Exception):
    """当某个类别的样本数量不足时抛出"""
    pass


class EpisodeSampler:
    """N-way K-shot Episode 采样器"""

    def __init__(
        self,
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 15
    ):
        """
        初始化 Episode 采样器
        
        Args:
            n_way: Episode 中的类别数量（默认为2：human vs bot）
            k_shot: 每个类别的 support 样本数量
            n_query: 每个类别的 query 样本数量
            
        Raises:
            ValueError: 当参数无效时
        """
        if n_way < 1:
            raise ValueError(f"Invalid episode config: n_way={n_way}, k_shot={k_shot}")
        if k_shot < 1:
            raise ValueError(f"Invalid episode config: n_way={n_way}, k_shot={k_shot}")
        if n_query < 1:
            raise ValueError(f"Invalid episode config: n_way={n_way}, k_shot={k_shot}")
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def sample(
        self,
        dataset: BotDataset,
        indices: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        从指定索引中采样一个 episode
        
        Args:
            dataset: BotDataset 实例
            indices: 可用于采样的索引张量
            
        Returns:
            support_set: {
                'num_features': Tensor[n_way*k_shot, 5],
                'cat_features': Tensor[n_way*k_shot, 3],
                'labels': Tensor[n_way*k_shot]
            }
            query_set: {
                'num_features': Tensor[n_way*n_query, 5],
                'cat_features': Tensor[n_way*n_query, 3],
                'labels': Tensor[n_way*n_query]
            }
            
        Raises:
            ValueError: 当索引集为空时
            InsufficientSamplesError: 当某类别样本不足时
        """
        if len(indices) == 0:
            raise ValueError("Cannot sample from empty index set")
        
        # 获取指定索引的标签
        labels = dataset.labels[indices]
        
        # 找出所有唯一的类别（排除 -1 即未知标签）
        unique_labels = torch.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        if len(valid_labels) < self.n_way:
            raise InsufficientSamplesError(
                f"Not enough classes: found {len(valid_labels)}, need {self.n_way}"
            )
        
        # 选择 n_way 个类别（对于二分类，通常是 0 和 1）
        selected_classes = valid_labels[:self.n_way].tolist()
        
        # 每个类别需要的样本数
        samples_needed = self.k_shot + self.n_query
        
        # 收集每个类别的索引
        class_indices = {}
        for label in selected_classes:
            # 找出属于该类别的样本在 indices 中的位置
            mask = labels == label
            class_idx = indices[mask]
            
            if len(class_idx) < samples_needed:
                raise InsufficientSamplesError(
                    f"Class {label} has {len(class_idx)} samples, need {samples_needed}"
                )
            
            class_indices[label] = class_idx
        
        # 采样 support 和 query
        support_num_features = []
        support_cat_features = []
        support_labels = []
        query_num_features = []
        query_cat_features = []
        query_labels = []
        
        for class_label in selected_classes:
            # 随机打乱该类别的索引
            class_idx = class_indices[class_label]
            perm = torch.randperm(len(class_idx))
            shuffled_idx = class_idx[perm]
            
            # 前 k_shot 个作为 support
            support_idx = shuffled_idx[:self.k_shot]
            # 接下来 n_query 个作为 query
            query_idx = shuffled_idx[self.k_shot:self.k_shot + self.n_query]
            
            # 收集 support 特征
            for idx in support_idx:
                item = dataset[idx.item()]
                support_num_features.append(item['num_features'])
                support_cat_features.append(item['cat_features'])
                support_labels.append(class_label)
            
            # 收集 query 特征
            for idx in query_idx:
                item = dataset[idx.item()]
                query_num_features.append(item['num_features'])
                query_cat_features.append(item['cat_features'])
                query_labels.append(class_label)
        
        # 构建返回字典
        support_set = {
            'num_features': torch.stack(support_num_features),
            'cat_features': torch.stack(support_cat_features),
            'labels': torch.tensor(support_labels, dtype=torch.long)
        }
        
        query_set = {
            'num_features': torch.stack(query_num_features),
            'cat_features': torch.stack(query_cat_features),
            'labels': torch.tensor(query_labels, dtype=torch.long)
        }
        
        return support_set, query_set
