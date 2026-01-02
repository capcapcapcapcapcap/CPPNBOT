"""
统一数据预处理系统
将Twibot-20和Misbot转换为统一格式，供原型网络使用

输出结构:
    processed_data/{dataset}/
    ├── num_features.pt    # 数值特征 [N, 8] (已标准化)
    ├── cat_features.pt    # 分类特征 [N, D] (D因数据集而异)
    ├── labels.pt          # 标签 [N] (0=human, 1=bot, -1=未标注)
    ├── train_idx.pt       # 训练集索引
    ├── val_idx.pt         # 验证集索引
    ├── test_idx.pt        # 测试集索引
    ├── edge_index.pt      # 图边 [2, E]
    ├── edge_type.pt       # 边类型 [E]
    ├── user_texts.json    # 用户文本 {idx: {description, tweets}}
    └── metadata.json      # 元数据

数值特征 (8维，统一):
    0. followers_count      - 粉丝数
    1. following_count      - 关注数
    2. tweet_count          - 推文数
    3. listed_count         - 被列表收录次数 (Misbot填0)
    4. account_age_days     - 账户年龄天数 (Misbot填0)
    5. followers_following_ratio - 粉丝/关注比 (防止除0)
    6. username_length      - 用户名长度
    7. description_length   - 简介长度 (两个数据集都有)

分类特征 (因数据集而异):
    Twibot-20 (5维): verified, protected, default_avatar, has_url, has_location
    Misbot (20维): 原始20维one-hot特征

使用方法:
    python preprocess_unified.py --dataset twibot20
    python preprocess_unified.py --dataset misbot
    python preprocess_unified.py --dataset all
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Generator

# 流式JSON解析
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    print("警告: ijson未安装，大文件将使用标准方式加载")
    print("建议安装: pip install ijson")


# ============== 配置 ==============
@dataclass
class PreprocessConfig:
    dataset_name: str
    input_dir: Path
    output_dir: Path
    max_tweets_per_user: int = 20  # 每用户最多保存的推文数


# ============== 流式JSON加载 ==============
class StreamingJsonLoader:
    """流式JSON加载器 - 解决大文件内存问题"""
    
    @staticmethod
    def iter_json_objects(filepath: Path, desc: str = "加载") -> Generator[Tuple[str, dict], None, None]:
        """流式迭代JSON对象"""
        if not IJSON_AVAILABLE:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key, value in tqdm(data.items(), desc=desc):
                yield key, value
            return
        
        with open(filepath, 'rb') as f:
            parser = ijson.kvitems(f, '', use_float=True)
            for key, value in tqdm(parser, desc=desc):
                yield key, value


# ============== 基础预处理器 ==============
class BasePreprocessor(ABC):
    """预处理器基类"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
        """加载数据，返回: user_df, tweet_df, labels, splits"""
        pass
    
    @abstractmethod
    def extract_numerical_features(self, user_df: pd.DataFrame) -> torch.Tensor:
        """提取8维数值特征"""
        pass
    
    @abstractmethod
    def extract_categorical_features(self, user_df: pd.DataFrame) -> torch.Tensor:
        """提取5维分类特征"""
        pass
    
    @abstractmethod
    def build_graph(self, user_df: pd.DataFrame, uid_to_idx: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建图结构，返回: edge_index, edge_type"""
        pass
    
    @abstractmethod
    def extract_user_texts(self, user_df: pd.DataFrame, tweet_df: pd.DataFrame, 
                           uid_to_idx: dict) -> Dict[int, dict]:
        """提取用户文本，返回: {user_idx: {description, tweets}}"""
        pass
    
    @abstractmethod
    def get_cat_feature_info(self) -> tuple:
        """返回分类特征信息: (维度, 特征名列表)"""
        pass
    
    def run(self):
        """执行预处理"""
        print(f"\n{'='*60}")
        print(f"预处理: {self.config.dataset_name}")
        print(f"{'='*60}")
        
        # 1. 加载数据
        print("\n[1/6] 加载数据...")
        user_df, tweet_df, labels, splits = self.load_data()
        print(f"  用户: {len(user_df)}, 推文: {len(tweet_df)}")
        
        # 2. 构建索引映射
        print("\n[2/6] 构建索引...")
        user_ids = list(user_df['id'])
        uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        
        # 3. 处理标签和划分
        print("\n[3/6] 处理标签和划分...")
        self._save_labels_and_splits(user_ids, labels, splits, uid_to_idx)
        
        # 4. 提取特征
        print("\n[4/6] 提取特征...")
        num_features = self.extract_numerical_features(user_df)
        num_features = self._standardize(num_features)
        torch.save(num_features, self.config.output_dir / "num_features.pt")
        print(f"  数值特征: {num_features.shape}")
        
        cat_features = self.extract_categorical_features(user_df)
        torch.save(cat_features, self.config.output_dir / "cat_features.pt")
        print(f"  分类特征: {cat_features.shape}")
        
        # 5. 构建图
        print("\n[5/6] 构建图...")
        edge_index, edge_type = self.build_graph(user_df, uid_to_idx)
        torch.save(edge_index, self.config.output_dir / "edge_index.pt")
        torch.save(edge_type, self.config.output_dir / "edge_type.pt")
        print(f"  边数量: {edge_index.shape[1]}")
        
        # 6. 提取文本
        print("\n[6/6] 提取文本...")
        user_texts = self.extract_user_texts(user_df, tweet_df, uid_to_idx)
        with open(self.config.output_dir / "user_texts.json", 'w', encoding='utf-8') as f:
            json.dump(user_texts, f, ensure_ascii=False)
        print(f"  用户文本: {len(user_texts)}")
        
        # 保存元数据
        cat_dim, cat_names = self.get_cat_feature_info()
        self._save_metadata(len(user_df), len(tweet_df), edge_index.shape[1], cat_dim, cat_names)
        
        print(f"\n完成! 输出: {self.config.output_dir}")
    
    def _save_labels_and_splits(self, user_ids, labels, splits, uid_to_idx):
        """保存标签和划分索引"""
        # 标签
        label_list = []
        for uid in user_ids:
            label = labels.get(uid, 'unknown')
            if label == 'human':
                label_list.append(0)
            elif label == 'bot':
                label_list.append(1)
            else:
                label_list.append(-1)
        
        torch.save(torch.tensor(label_list, dtype=torch.long), 
                   self.config.output_dir / "labels.pt")
        
        # 划分
        train_idx = [uid_to_idx[uid] for uid in splits.get('train', []) if uid in uid_to_idx]
        val_idx = [uid_to_idx[uid] for uid in splits.get('val', splits.get('dev', [])) if uid in uid_to_idx]
        test_idx = [uid_to_idx[uid] for uid in splits.get('test', []) if uid in uid_to_idx]
        
        torch.save(torch.tensor(train_idx, dtype=torch.long), self.config.output_dir / "train_idx.pt")
        torch.save(torch.tensor(val_idx, dtype=torch.long), self.config.output_dir / "val_idx.pt")
        torch.save(torch.tensor(test_idx, dtype=torch.long), self.config.output_dir / "test_idx.pt")
        
        print(f"  标签分布: human={label_list.count(0)}, bot={label_list.count(1)}, unknown={label_list.count(-1)}")
        print(f"  划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    def _standardize(self, features: torch.Tensor) -> torch.Tensor:
        """Z-score标准化"""
        features = features.float()
        for i in range(features.shape[1]):
            col = features[:, i]
            mean, std = col.mean(), col.std()
            if std > 0:
                features[:, i] = (col - mean) / std
        return features
    
    def _save_metadata(self, n_users, n_tweets, n_edges, cat_feature_dim, cat_feature_names):
        """保存元数据"""
        metadata = {
            "dataset": self.config.dataset_name,
            "n_users": n_users,
            "n_tweets": n_tweets,
            "n_edges": n_edges,
            "num_feature_dim": 8,
            "num_feature_names": [
                "followers_count",
                "following_count", 
                "tweet_count",
                "listed_count",
                "account_age_days",
                "followers_following_ratio",
                "username_length",
                "description_length"
            ],
            "cat_feature_dim": cat_feature_dim,
            "cat_feature_names": cat_feature_names,
            "processed_time": dt.now().isoformat()
        }
        with open(self.config.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)



# ============== Twibot-20 预处理器 ==============
class Twibot20Preprocessor(BasePreprocessor):
    """Twibot-20数据集预处理器 - 低内存版本"""
    
    def get_cat_feature_info(self) -> tuple:
        """返回分类特征信息"""
        return 5, ["verified", "protected", "default_avatar", "has_url", "has_location"]
    
    def load_data(self):
        """流式加载Twibot-20数据 (只加载用户，推文延迟加载)"""
        print("  流式加载node.json (只加载用户)...")
        
        users = []
        self._tweet_ids = []  # 只存ID，不存内容
        user_fields = ['id', 'username', 'name', 'description', 'created_at', 
                       'public_metrics', 'profile_image_url', 'verified', 'protected',
                       'url', 'location']
        
        for node_id, node_data in StreamingJsonLoader.iter_json_objects(
            self.config.input_dir / "node.json", "解析节点"
        ):
            if node_id.startswith('u'):
                record = {'id': node_id}
                for field in user_fields:
                    if field in node_data:
                        record[field] = node_data[field]
                users.append(record)
            elif node_id.startswith('t'):
                self._tweet_ids.append(node_id)
        
        user_df = pd.DataFrame(users)
        # 返回空的tweet_df，后续按需加载
        tweet_df = pd.DataFrame({'id': self._tweet_ids})
        
        print(f"  用户: {len(users)}, 推文ID: {len(self._tweet_ids)}")
        
        # 加载标签和划分 (小文件)
        with open(self.config.input_dir / "label.json", 'r') as f:
            label_data = json.load(f)
        labels = {k: v for k, v in label_data.items() if k != 'id'}
        
        with open(self.config.input_dir / "split.json", 'r') as f:
            splits = json.load(f)
        
        return user_df, tweet_df, labels, splits
    
    def extract_numerical_features(self, user_df: pd.DataFrame) -> torch.Tensor:
        """提取8维数值特征"""
        features = []
        reference_date = dt(2020, 9, 1)
        
        for _, row in user_df.iterrows():
            metrics = row.get('public_metrics', {}) or {}
            
            # 1. followers_count
            followers = metrics.get('followers_count', 0) or 0
            # 2. following_count
            following = metrics.get('following_count', 0) or 0
            # 3. tweet_count
            tweets = metrics.get('tweet_count', 0) or 0
            # 4. listed_count
            listed = metrics.get('listed_count', 0) or 0
            # 5. account_age_days
            try:
                created_at = row.get('created_at')
                if created_at:
                    created = pd.to_datetime(created_at)
                    if created.tzinfo:
                        created = created.tz_localize(None)
                    age = (reference_date - created).days
                    age = max(0, age)
                else:
                    age = 0
            except:
                age = 0
            # 6. followers_following_ratio (防止除0)
            ratio = followers / (following + 1)
            # 7. username_length
            username = row.get('username', '') or ''
            username_len = len(username)
            # 8. description_length
            description = row.get('description', '') or ''
            desc_len = len(description)
            
            features.append([followers, following, tweets, listed, age, ratio, username_len, desc_len])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_categorical_features(self, user_df: pd.DataFrame) -> torch.Tensor:
        """提取5维分类特征: [verified, protected, default_avatar, has_url, has_location]"""
        features = []
        default_avatar_url = 'http://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'
        
        for _, row in user_df.iterrows():
            # 1. verified
            verified = 1 if row.get('verified') == True else 0
            # 2. protected
            protected = 1 if row.get('protected') == True else 0
            # 3. default_avatar
            img_url = row.get('profile_image_url', '') or ''
            default_avatar = 1 if img_url.strip() == default_avatar_url else 0
            # 4. has_url
            url = row.get('url', '') or ''
            has_url = 1 if url.strip() else 0
            # 5. has_location
            location = row.get('location', '') or ''
            has_location = 1 if location.strip() else 0
            
            features.append([verified, protected, default_avatar, has_url, has_location])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def build_graph(self, user_df, uid_to_idx):
        """流式构建图结构 (不缓存所有边到内存)"""
        print("  流式处理edge.json...")
        
        edge_index, edge_type = [], []
        
        # 直接流式处理，不存储中间结果
        for source_id, relations in StreamingJsonLoader.iter_json_objects(
            self.config.input_dir / "edge.json", "构建图"
        ):
            if source_id == 'source_id' or source_id not in uid_to_idx:
                continue
            
            if not isinstance(relations, list):
                continue
                
            for rel in relations:
                if not isinstance(rel, list) or len(rel) < 2:
                    continue
                    
                relation, target_id = rel[0], rel[1]
                
                # 跳过post关系
                if relation == 'post':
                    continue
                
                if target_id not in uid_to_idx:
                    continue
                
                edge_index.append([uid_to_idx[source_id], uid_to_idx[target_id]])
                edge_type.append(0 if relation == 'friend' else 1)
        
        if edge_index:
            return (torch.tensor(edge_index, dtype=torch.long).t(),
                    torch.tensor(edge_type, dtype=torch.long))
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    def extract_user_texts(self, user_df, tweet_df, uid_to_idx):
        """流式提取用户文本 (分两次遍历，避免内存爆炸)"""
        
        # 第一遍：从edge.json获取用户-推文映射
        print("  构建用户-推文映射...")
        user_tweet_ids = {i: [] for i in range(len(uid_to_idx))}
        
        for source_id, relations in StreamingJsonLoader.iter_json_objects(
            self.config.input_dir / "edge.json", "扫描post关系"
        ):
            if source_id == 'source_id' or source_id not in uid_to_idx:
                continue
            
            if not isinstance(relations, list):
                continue
            
            user_idx = uid_to_idx[source_id]
            for rel in relations:
                if isinstance(rel, list) and len(rel) >= 2 and rel[0] == 'post':
                    if len(user_tweet_ids[user_idx]) < self.config.max_tweets_per_user:
                        user_tweet_ids[user_idx].append(rel[1])
        
        # 收集需要的推文ID
        needed_tweet_ids = set()
        for tweet_ids in user_tweet_ids.values():
            needed_tweet_ids.update(tweet_ids)
        print(f"  需要加载的推文数: {len(needed_tweet_ids)}")
        
        # 第二遍：从node.json获取推文内容 (只加载需要的)
        print("  加载推文内容...")
        tid_to_text = {}
        for node_id, node_data in StreamingJsonLoader.iter_json_objects(
            self.config.input_dir / "node.json", "加载推文"
        ):
            if node_id in needed_tweet_ids:
                tid_to_text[node_id] = node_data.get('text', '')
                if len(tid_to_text) >= len(needed_tweet_ids):
                    break  # 已经找到所有需要的推文
        
        # 构建最终结果
        print("  组装用户文本...")
        user_texts = {}
        for _, row in tqdm(user_df.iterrows(), total=len(user_df), desc="组装"):
            idx = uid_to_idx[row['id']]
            description = row.get('description', '') or ''
            
            tweet_ids = user_tweet_ids[idx]
            tweets = [tid_to_text.get(tid, '') for tid in tweet_ids]
            tweets = [t for t in tweets if t]
            
            user_texts[str(idx)] = {
                'description': description,
                'tweets': tweets
            }
        
        return user_texts



# ============== Misbot 预处理器 ==============
class MisbotPreprocessor(BasePreprocessor):
    """Misbot数据集预处理器"""
    
    def get_cat_feature_info(self) -> tuple:
        """返回分类特征信息"""
        return 20, [f"cat_{i}" for i in range(20)]
    
    def load_data(self):
        """加载Misbot数据"""
        print("  加载node.json (字典格式)...")
        
        # Misbot的node.json是字典格式: {node_id: {属性...}, ...}
        users, tweets = [], []
        with open(self.config.input_dir / "node.json", 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        
        for node_id, node_data in tqdm(nodes.items(), desc="解析节点"):
            node_data['id'] = node_id  # 统一用小写id
            if node_data.get('type') == 'user' or not node_id.startswith('t_'):
                users.append(node_data)
            else:
                tweets.append(node_data)
        
        user_df = pd.DataFrame(users)
        tweet_df = pd.DataFrame(tweets)
        
        print(f"  用户列: {list(user_df.columns)[:5]}...")
        
        # 加载边
        with open(self.config.input_dir / "edge.json", 'r', encoding='utf-8') as f:
            self._edge_data = json.load(f)
        
        # 加载标签
        with open(self.config.input_dir / "label.json", 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        labels = {k: v for k, v in label_data.items() if k != 'id'}
        
        # 加载划分
        with open(self.config.input_dir / "split.json", 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        splits = {
            'train': split_data.get('train', []),
            'val': split_data.get('dev', []),
            'test': split_data.get('test', [])
        }
        
        return user_df, tweet_df, labels, splits
    
    def extract_numerical_features(self, user_df: pd.DataFrame) -> torch.Tensor:
        """提取8维数值特征 (Misbot缺少部分特征，填0)"""
        features = []
        
        for _, row in user_df.iterrows():
            profile = row.get('profile', {}) or {}
            numerical = profile.get('numerical', [0, 0, 0])
            description = profile.get('description', '') or ''
            
            if len(numerical) >= 3:
                # Misbot numerical: [followers, following, tweets]
                followers = numerical[0] if numerical[0] is not None else 0
                following = numerical[1] if numerical[1] is not None else 0
                tweets = numerical[2] if numerical[2] is not None else 0
            else:
                followers, following, tweets = 0, 0, 0
            
            # 1. followers_count
            # 2. following_count
            # 3. tweet_count
            # 4. listed_count (Misbot没有，填0)
            listed = 0
            # 5. account_age_days (Misbot没有，填0)
            age = 0
            # 6. followers_following_ratio (防止除0)
            ratio = followers / (following + 1)
            # 7. username_length (从ID提取，去掉前缀)
            uid = row.get('id', '') or ''
            # 去掉 "train_u" 或类似前缀
            username_len = len(uid.replace('train_', '').replace('test_', '').replace('dev_', ''))
            # 8. description_length (两个数据集都有)
            desc_len = len(description)
            
            features.append([followers, following, tweets, listed, age, ratio, username_len, desc_len])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_categorical_features(self, user_df: pd.DataFrame) -> torch.Tensor:
        """提取20维分类特征 (保留Misbot完整的categorical向量)"""
        features = []
        
        for _, row in user_df.iterrows():
            profile = row.get('profile', {}) or {}
            categorical = profile.get('categorical', [])
            
            # Misbot的categorical是20维one-hot向量，全部保留
            if len(categorical) >= 20:
                feat = [int(categorical[i]) if categorical[i] is not None else 0 for i in range(20)]
            else:
                # 如果不足20维，用0填充
                feat = [int(categorical[i]) if i < len(categorical) and categorical[i] is not None else 0 for i in range(20)]
            
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def build_graph(self, user_df, uid_to_idx):
        """构建图结构"""
        edge_index, edge_type = [], []
        
        for source_id, relations in tqdm(self._edge_data.items(), desc="构建图"):
            if source_id == 'id' or source_id not in uid_to_idx:
                continue
            
            if not isinstance(relations, list):
                continue
            
            for rel in relations:
                if not isinstance(rel, list) or len(rel) < 2:
                    continue
                
                relation, target_id = rel[0], rel[1]
                
                # 跳过post关系
                if relation == 'post':
                    continue
                
                if target_id not in uid_to_idx:
                    continue
                
                edge_index.append([uid_to_idx[source_id], uid_to_idx[target_id]])
                edge_type.append(0 if relation in ['friend', 'follow'] else 1)
        
        if edge_index:
            return (torch.tensor(edge_index, dtype=torch.long).t(),
                    torch.tensor(edge_type, dtype=torch.long))
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    def extract_user_texts(self, user_df, tweet_df, uid_to_idx):
        """提取用户文本"""
        # 构建推文索引
        tid_to_text = {}
        for _, row in tweet_df.iterrows():
            tid = row['id']
            text = row.get('content', row.get('text', '')) or ''
            tid_to_text[tid] = text
        
        # 提取文本
        user_texts = {}
        for _, row in tqdm(user_df.iterrows(), total=len(user_df), desc="提取文本"):
            uid = row['id']
            idx = uid_to_idx[uid]
            
            # 描述
            profile = row.get('profile', {}) or {}
            description = profile.get('description', '') or ''
            
            # 推文
            tweets = []
            relations = self._edge_data.get(uid, [])
            if isinstance(relations, list):
                for rel in relations:
                    if isinstance(rel, list) and len(rel) >= 2 and rel[0] == 'post':
                        tid = rel[1]
                        if tid in tid_to_text and tid_to_text[tid]:
                            tweets.append(tid_to_text[tid])
                            if len(tweets) >= self.config.max_tweets_per_user:
                                break
            
            user_texts[str(idx)] = {
                'description': description,
                'tweets': tweets
            }
        
        return user_texts


# ============== 统一预处理管理器 ==============
class UnifiedPreprocessor:
    """统一预处理管理器"""
    
    DATASETS = {
        "twibot20": {
            "class": Twibot20Preprocessor,
            "input_dir": Path("./dataset/Twibot-20"),
            "output_dir": Path("./processed_data/twibot20"),
        },
        "misbot": {
            "class": MisbotPreprocessor,
            "input_dir": Path("./dataset/Misbot"),
            "output_dir": Path("./processed_data/misbot"),
        }
    }
    
    def preprocess(self, dataset_name: str):
        """预处理单个数据集"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        info = self.DATASETS[dataset_name]
        config = PreprocessConfig(
            dataset_name=dataset_name,
            input_dir=info["input_dir"],
            output_dir=info["output_dir"]
        )
        
        preprocessor = info["class"](config)
        preprocessor.run()
    
    def preprocess_all(self):
        """预处理所有数据集"""
        for name in self.DATASETS:
            try:
                self.preprocess(name)
            except Exception as e:
                print(f"预处理 {name} 失败: {e}")
                import traceback
                traceback.print_exc()


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="统一数据预处理")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["twibot20", "misbot", "all"],
                        help="要预处理的数据集")
    args = parser.parse_args()
    
    preprocessor = UnifiedPreprocessor()
    
    if args.dataset == "all":
        preprocessor.preprocess_all()
    else:
        preprocessor.preprocess(args.dataset)


if __name__ == "__main__":
    main()
