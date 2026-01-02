"""
BotDataset: 加载预处理后的机器人检测数据集

Implements Requirements 1.1, 1.2, 1.3, 1.4

支持预计算的文本嵌入以加速训练。
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch


class BotDataset:
    """加载预处理后的机器人检测数据集
    
    支持两种文本模式:
    1. 原始文本模式: 加载 user_texts.json，训练时在线编码
    2. 预计算嵌入模式: 加载 text_embeddings.pt，直接使用预计算的嵌入
    """

    def __init__(self, dataset_name: str, data_dir: str = "processed_data"):
        """
        初始化数据集，加载所有预处理张量
        
        Args:
            dataset_name: 'twibot20' 或 'misbot'
            data_dir: 预处理数据根目录
            
        Raises:
            FileNotFoundError: 当数据集目录或必需文件不存在时
        """
        self.dataset_name = dataset_name
        self.data_path = Path(data_dir) / dataset_name
        
        # 验证数据集目录存在
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_path}")
        
        # 加载所有必需的张量文件
        self.num_features = self._load_tensor("num_features.pt")
        self.cat_features = self._load_tensor("cat_features.pt")
        self.labels = self._load_tensor("labels.pt")
        self.edge_index = self._load_tensor("edge_index.pt")
        self.edge_type = self._load_tensor("edge_type.pt")
        
        # 加载划分索引
        self.train_idx = self._load_tensor("train_idx.pt")
        self.val_idx = self._load_tensor("val_idx.pt")
        self.test_idx = self._load_tensor("test_idx.pt")
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        # 缓存用户文本（延迟加载）
        self._user_texts: Optional[Dict[int, str]] = None
        
        # 预计算的文本嵌入（延迟加载）
        self._text_embeddings: Optional[torch.Tensor] = None
        self._has_precomputed_embeddings = (self.data_path / "text_embeddings.pt").exists()

    def _load_tensor(self, filename: str) -> torch.Tensor:
        """
        加载单个张量文件
        
        Args:
            filename: 张量文件名
            
        Returns:
            加载的张量
            
        Raises:
            FileNotFoundError: 当文件不存在时
            RuntimeError: 当张量加载失败时
        """
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filename}")
        
        try:
            return torch.load(filepath, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load tensor: {filename}") from e

    def _load_metadata(self) -> Dict:
        """
        加载数据集元数据
        
        Returns:
            元数据字典
            
        Raises:
            FileNotFoundError: 当元数据文件不存在时
            json.JSONDecodeError: 当JSON格式无效时
        """
        metadata_path = self.data_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Required file not found: metadata.json")
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError("Invalid metadata format", e.doc, e.pos)

    def __len__(self) -> int:
        """返回数据集中用户数量"""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回单个用户的特征
        
        Args:
            idx: 用户索引
            
        Returns:
            包含 num_features, cat_features, label 的字典
        """
        return {
            'num_features': self.num_features[idx],
            'cat_features': self.cat_features[idx],
            'label': self.labels[idx]
        }

    def get_split_indices(self, split: str) -> torch.Tensor:
        """
        获取 train/val/test 划分的索引
        
        Args:
            split: 'train', 'val', 或 'test'
            
        Returns:
            对应划分的索引张量
            
        Raises:
            ValueError: 当 split 参数无效时
        """
        if split == 'train':
            return self.train_idx
        elif split == 'val':
            return self.val_idx
        elif split == 'test':
            return self.test_idx
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    def get_user_texts(self) -> Dict[int, str]:
        """
        加载用户文本数据（描述+推文）
        
        Returns:
            用户索引到文本内容的映射
            
        Raises:
            FileNotFoundError: 当文本文件不存在时
            json.JSONDecodeError: 当JSON格式无效时
        """
        if self._user_texts is not None:
            return self._user_texts
        
        texts_path = self.data_path / "user_texts.json"
        if not texts_path.exists():
            raise FileNotFoundError(f"Required file not found: user_texts.json")
        
        try:
            with open(texts_path, 'r', encoding='utf-8') as f:
                # JSON keys are strings, convert to int
                raw_texts = json.load(f)
                self._user_texts = {int(k): v for k, v in raw_texts.items()}
                return self._user_texts
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError("Invalid metadata format", e.doc, e.pos)

    def has_precomputed_text_embeddings(self) -> bool:
        """检查是否有预计算的文本嵌入"""
        return self._has_precomputed_embeddings
    
    def get_text_embeddings(self) -> torch.Tensor:
        """
        加载预计算的文本嵌入
        
        Returns:
            Tensor[num_users, embed_dim] 预计算的文本嵌入
            
        Raises:
            FileNotFoundError: 当嵌入文件不存在时
        """
        if self._text_embeddings is not None:
            return self._text_embeddings
        
        embeddings_path = self.data_path / "text_embeddings.pt"
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Precomputed text embeddings not found: {embeddings_path}\n"
                f"Run: python precompute_text_embeddings.py --dataset {self.dataset_name}"
            )
        
        self._text_embeddings = torch.load(embeddings_path, weights_only=True)
        return self._text_embeddings
    
    def get_text_embeddings_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        获取指定索引的预计算文本嵌入
        
        Args:
            indices: 用户索引张量
            
        Returns:
            Tensor[len(indices), embed_dim] 对应的文本嵌入
        """
        embeddings = self.get_text_embeddings()
        return embeddings[indices]
