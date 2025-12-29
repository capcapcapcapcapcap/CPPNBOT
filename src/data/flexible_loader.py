#!/usr/bin/env python3
"""
灵活数据加载器
支持加载预处理后的多平台数据，用于元学习训练
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import yaml

@dataclass
class LoadedPlatformData:
    """加载后的平台数据"""
    user_ids: List[str]
    text_features: torch.Tensor          # [N, text_dim]
    numerical_features: Optional[torch.Tensor] = None  # [N, num_dim]
    categorical_features: Optional[torch.Tensor] = None # [N, cat_dim]
    edge_index: Optional[torch.Tensor] = None          # [2, E]
    edge_types: Optional[torch.Tensor] = None          # [E]
    labels: torch.Tensor                 # [N]
    
    platform_name: str = ""
    language: str = ""
    feature_description: Dict = None
    
    def __len__(self):
        return len(self.user_ids)
    
    def to(self, device):
        """移动到指定设备"""
        self.text_features = self.text_features.to(device)
        if self.numerical_features is not None:
            self.numerical_features = self.numerical_features.to(device)
        if self.categorical_features is not None:
            self.categorical_features = self.categorical_features.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_types is not None:
            self.edge_types = self.edge_types.to(device)
        self.labels = self.labels.to(device)
        return self

class FlexibleDataLoader:
    """灵活数据加载器 - 支持多平台预处理数据"""
    
    def __init__(self, processed_data_dir: str = "processed_data"):
        self.data_dir = Path(processed_data_dir)
        self.available_datasets = self._discover_datasets()
        
    def _discover_datasets(self) -> Dict[str, Dict]:
        """发现可用的数据集"""
        datasets = {}
        
        # 查找元数据文件
        for meta_file in self.data_dir.glob("*_metadata.json"):
            dataset_name = meta_file.stem.replace("_metadata", "")
            
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            datasets[dataset_name] = {
                'metadata_file': str(meta_file),
                'metadata': metadata,
                'files': metadata.get('saved_files', {})
            }
        
        return datasets
    
    def list_available_datasets(self) -> List[str]:
        """列出可用的数据集"""
        return list(self.available_datasets.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {self.list_available_datasets()}")
        
        return self.available_datasets[dataset_name]['metadata']
    
    def load_dataset(self, dataset_name: str, device: str = "cpu") -> LoadedPlatformData:
        """加载指定数据集"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {self.list_available_datasets()}")
        
        dataset_info = self.available_datasets[dataset_name]
        files = dataset_info['files']
        metadata = dataset_info['metadata']
        
        print(f"Loading {dataset_name} dataset...")
        
        # 加载用户ID
        with open(files['user_ids'], 'r') as f:
            user_ids = json.load(f)
        
        # 加载文本特征（必须）
        text_features = torch.load(files['text_features'], map_location=device)
        
        # 加载标签（必须）
        labels = torch.load(files['labels'], map_location=device)
        
        # 加载可选特征
        numerical_features = None
        if 'numerical_features' in files:
            numerical_features = torch.load(files['numerical_features'], map_location=device)
        
        categorical_features = None
        if 'categorical_features' in files:
            categorical_features = torch.load(files['categorical_features'], map_location=device)
        
        edge_index = None
        edge_types = None
        if 'edge_index' in files and 'edge_types' in files:
            edge_index = torch.load(files['edge_index'], map_location=device)
            edge_types = torch.load(files['edge_types'], map_location=device)
        
        return LoadedPlatformData(
            user_ids=user_ids,
            text_features=text_features,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            edge_index=edge_index,
            edge_types=edge_types,
            labels=labels,
            platform_name=metadata.get('platform_name', dataset_name),
            language=metadata.get('language', 'unknown'),
            feature_description=metadata.get('feature_description', {})
        )
    
    def load_twibot20(self, device: str = "cpu") -> LoadedPlatformData:
        """加载Twibot-20数据集"""
        return self.load_dataset('twibot20', device)
    
    def load_misbot(self, device: str = "cpu") -> LoadedPlatformData:
        """加载Misbot数据集"""
        return self.load_dataset('misbot', device)
    
    def load_both_datasets(self, device: str = "cpu") -> Tuple[LoadedPlatformData, LoadedPlatformData]:
        """同时加载两个数据集"""
        twibot_data = self.load_twibot20(device)
        misbot_data = self.load_misbot(device)
        return twibot_data, misbot_data
    
    def get_feature_compatibility_info(self) -> Dict:
        """获取特征兼容性信息"""
        compatibility = {
            'datasets': list(self.available_datasets.keys()),
            'feature_dimensions': {},
            'available_modalities': {},
            'compatibility_matrix': {}
        }
        
        for dataset_name, dataset_info in self.available_datasets.items():
            metadata = dataset_info['metadata']
            files = dataset_info['files']
            
            # 特征维度
            compatibility['feature_dimensions'][dataset_name] = {}
            
            if 'text_features' in files:
                text_features = torch.load(files['text_features'], map_location='cpu')
                compatibility['feature_dimensions'][dataset_name]['text'] = text_features.shape[1]
            
            if 'numerical_features' in files:
                num_features = torch.load(files['numerical_features'], map_location='cpu')
                compatibility['feature_dimensions'][dataset_name]['numerical'] = num_features.shape[1]
            
            if 'categorical_features' in files:
                cat_features = torch.load(files['categorical_features'], map_location='cpu')
                compatibility['feature_dimensions'][dataset_name]['categorical'] = cat_features.shape[1]
            
            # 可用模态
            compatibility['available_modalities'][dataset_name] = {
                'text': 'text_features' in files,
                'numerical': 'numerical_features' in files,
                'categorical': 'categorical_features' in files,
                'graph': 'edge_index' in files and 'edge_types' in files
            }
        
        return compatibility
    
    def create_unified_batch(self, datasets: List[LoadedPlatformData], 
                           indices_list: List[List[int]]) -> Dict[str, torch.Tensor]:
        """创建统一的批次数据（用于跨域训练）"""
        batch = {
            'text_features': [],
            'numerical_features': [],
            'categorical_features': [],
            'labels': [],
            'dataset_ids': [],
            'user_indices': []
        }
        
        for dataset_idx, (dataset, indices) in enumerate(zip(datasets, indices_list)):
            # 文本特征（必须有）
            batch['text_features'].append(dataset.text_features[indices])
            
            # 数值特征（可能缺失）
            if dataset.numerical_features is not None:
                batch['numerical_features'].append(dataset.numerical_features[indices])
            else:
                # 用零填充
                zero_num = torch.zeros(len(indices), 5, device=dataset.text_features.device)
                batch['numerical_features'].append(zero_num)
            
            # 分类特征（可能缺失）
            if dataset.categorical_features is not None:
                batch['categorical_features'].append(dataset.categorical_features[indices])
            else:
                # 用零填充
                zero_cat = torch.zeros(len(indices), 3, device=dataset.text_features.device)
                batch['categorical_features'].append(zero_cat)
            
            # 标签
            batch['labels'].append(dataset.labels[indices])
            
            # 数据集ID
            dataset_ids = torch.full((len(indices),), dataset_idx, 
                                   device=dataset.text_features.device)
            batch['dataset_ids'].append(dataset_ids)
            
            # 用户索引
            user_indices = torch.tensor(indices, device=dataset.text_features.device)
            batch['user_indices'].append(user_indices)
        
        # 拼接所有批次
        for key in ['text_features', 'numerical_features', 'categorical_features', 
                   'labels', 'dataset_ids', 'user_indices']:
            batch[key] = torch.cat(batch[key], dim=0)
        
        return batch

class EpisodeDataLoader:
    """Episode数据加载器 - 用于元学习训练"""
    
    def __init__(self, flexible_loader: FlexibleDataLoader):
        self.flexible_loader = flexible_loader
        self.datasets = {}
        
    def register_dataset(self, name: str, dataset: LoadedPlatformData):
        """注册数据集"""
        self.datasets[name] = dataset
        
    def register_all_datasets(self, device: str = "cpu"):
        """注册所有可用数据集"""
        for dataset_name in self.flexible_loader.list_available_datasets():
            dataset = self.flexible_loader.load_dataset(dataset_name, device)
            self.register_dataset(dataset_name, dataset)
    
    def sample_episode(self, source_dataset: str, n_way: int = 2, 
                      k_shot: int = 5, q_query: int = 15) -> Tuple[Dict, Dict]:
        """从指定数据集采样一个episode"""
        if source_dataset not in self.datasets:
            raise ValueError(f"Dataset {source_dataset} not registered")
        
        dataset = self.datasets[source_dataset]
        
        # 分离不同类别的样本
        human_indices = (dataset.labels == 0).nonzero(as_tuple=True)[0].tolist()
        bot_indices = (dataset.labels == 1).nonzero(as_tuple=True)[0].tolist()
        
        if len(human_indices) < k_shot + q_query or len(bot_indices) < k_shot + q_query:
            raise ValueError(f"Not enough samples for {k_shot}-shot {q_query}-query episode")
        
        # 随机采样
        import random
        
        # 支持集
        support_human = random.sample(human_indices, k_shot)
        support_bot = random.sample(bot_indices, k_shot)
        support_indices = support_human + support_bot
        
        # 查询集（排除支持集）
        remaining_human = [i for i in human_indices if i not in support_human]
        remaining_bot = [i for i in bot_indices if i not in support_bot]
        
        query_human = random.sample(remaining_human, q_query)
        query_bot = random.sample(remaining_bot, q_query)
        query_indices = query_human + query_bot
        
        # 构建episode数据
        support_data = self._build_episode_data(dataset, support_indices)
        query_data = self._build_episode_data(dataset, query_indices)
        
        return support_data, query_data
    
    def sample_cross_domain_episode(self, source_dataset: str, target_dataset: str,
                                   k_shot: int = 5, q_query: int = 15) -> Tuple[Dict, Dict]:
        """采样跨域episode"""
        # 从源域采样支持集
        support_data, _ = self.sample_episode(source_dataset, k_shot=k_shot, q_query=0)
        
        # 从目标域采样查询集
        _, query_data = self.sample_episode(target_dataset, k_shot=0, q_query=q_query)
        
        return support_data, query_data
    
    def _build_episode_data(self, dataset: LoadedPlatformData, 
                           indices: List[int]) -> Dict[str, torch.Tensor]:
        """构建episode数据"""
        episode_data = {
            'text_features': dataset.text_features[indices],
            'labels': dataset.labels[indices],
            'indices': torch.tensor(indices, device=dataset.text_features.device)
        }
        
        # 添加可选特征
        if dataset.numerical_features is not None:
            episode_data['numerical_features'] = dataset.numerical_features[indices]
        
        if dataset.categorical_features is not None:
            episode_data['categorical_features'] = dataset.categorical_features[indices]
        
        # 图结构需要特殊处理（子图提取）
        if dataset.edge_index is not None:
            subgraph_data = self._extract_subgraph(dataset, indices)
            episode_data.update(subgraph_data)
        
        return episode_data
    
    def _extract_subgraph(self, dataset: LoadedPlatformData, 
                         node_indices: List[int]) -> Dict[str, torch.Tensor]:
        """提取子图"""
        # 简化版本：只保留节点间的边
        node_set = set(node_indices)
        
        # 找到子图中的边
        edge_mask = []
        for i in range(dataset.edge_index.shape[1]):
            src, dst = dataset.edge_index[0, i].item(), dataset.edge_index[1, i].item()
            if src in node_set and dst in node_set:
                edge_mask.append(i)
        
        if edge_mask:
            subgraph_edge_index = dataset.edge_index[:, edge_mask]
            subgraph_edge_types = dataset.edge_types[edge_mask]
            
            # 重新映射节点索引
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
            
            remapped_edges = torch.zeros_like(subgraph_edge_index)
            for i in range(subgraph_edge_index.shape[1]):
                src, dst = subgraph_edge_index[0, i].item(), subgraph_edge_index[1, i].item()
                remapped_edges[0, i] = node_mapping[src]
                remapped_edges[1, i] = node_mapping[dst]
            
            return {
                'edge_index': remapped_edges,
                'edge_types': subgraph_edge_types
            }
        else:
            # 没有边的情况
            return {
                'edge_index': torch.empty((2, 0), dtype=torch.long, device=dataset.edge_index.device),
                'edge_types': torch.empty((0,), dtype=torch.long, device=dataset.edge_types.device)
            }

# 使用示例
def main():
    """使用示例"""
    
    # 1. 创建数据加载器
    loader = FlexibleDataLoader("processed_data")
    
    # 2. 查看可用数据集
    print("Available datasets:", loader.list_available_datasets())
    
    # 3. 查看特征兼容性
    compatibility = loader.get_feature_compatibility_info()
    print("Feature compatibility:", compatibility)
    
    # 4. 加载数据集
    if 'twibot20' in loader.list_available_datasets():
        twibot_data = loader.load_twibot20()
        print(f"Twibot-20: {len(twibot_data)} users")
        print(f"Text features shape: {twibot_data.text_features.shape}")
    
    if 'misbot' in loader.list_available_datasets():
        misbot_data = loader.load_misbot()
        print(f"Misbot: {len(misbot_data)} users")
        print(f"Text features shape: {misbot_data.text_features.shape}")
    
    # 5. Episode采样示例
    episode_loader = EpisodeDataLoader(loader)
    episode_loader.register_all_datasets()
    
    if 'twibot20' in episode_loader.datasets:
        support, query = episode_loader.sample_episode('twibot20', k_shot=5, q_query=15)
        print(f"Episode - Support: {support['labels'].shape}, Query: {query['labels'].shape}")

if __name__ == "__main__":
    main()