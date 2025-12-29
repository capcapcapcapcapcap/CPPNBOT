#!/usr/bin/env python3
"""
统一数据预处理器
结合Twibot作者的经验与我们的跨平台设计理念
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
from transformers import pipeline
import json
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PlatformData:
    """跨平台数据的统一接口"""
    
    # 必须字段
    user_ids: List[str]                    # 用户唯一标识
    user_texts: List[str]                  # 用户文本内容
    labels: Dict[str, int]                 # 用户标签 (0=human, 1=bot)
    
    # 可选字段（缺失时模型自动处理）
    numerical_features: Optional[np.ndarray] = None    # 数值特征
    graph_edges: Optional[List[Tuple]] = None          # 图边关系
    categorical_features: Optional[np.ndarray] = None  # 分类特征
    
    # 元信息
    platform_name: str = ""               # 平台标识
    language: str = ""                     # 主要语言
    feature_description: Dict = None       # 特征描述

class UnifiedPreprocessor:
    """统一预处理器 - 融合Twibot经验与跨平台设计"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'processed_data'))
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化文本编码器（多语言支持）
        model_name = config.get('text_model', 'xlm-roberta-base')
        self.text_encoder = pipeline(
            'feature-extraction',
            model=model_name,
            tokenizer=model_name,
            padding=True,
            truncation=True,
            max_length=config.get('max_text_length', 128),
            add_special_tokens=True
        )
        
        # 特征维度配置
        self.text_dim = config.get('text_dim', 768)
        self.max_numerical_features = config.get('max_numerical_features', 10)
        self.max_categorical_features = config.get('max_categorical_features', 5)
        
    def process_twibot20(self) -> PlatformData:
        """处理Twibot-20数据集（借鉴原作者经验）"""
        print("Processing Twibot-20 dataset...")
        
        # 1. 加载原始数据
        node_df = pd.read_json("./dataset/Twibot-20/node.json")
        edge_df = pd.read_json("./dataset/Twibot-20/edge.json")
        label_df = pd.read_json("./dataset/Twibot-20/label.json")
        split_df = pd.read_json("./dataset/Twibot-20/split.json")
        
        # 2. 分离用户和推文（借鉴fast_merge逻辑）
        user_data, tweet_data = self._split_user_tweet_twibot(node_df)
        
        # 3. 提取文本特征
        user_texts = self._extract_text_features_twibot(user_data, tweet_data, edge_df)
        
        # 4. 提取数值特征（借鉴原作者的5维特征）
        numerical_features = self._extract_numerical_features_twibot(user_data)
        
        # 5. 提取分类特征
        categorical_features = self._extract_categorical_features_twibot(user_data)
        
        # 6. 提取图结构（借鉴原作者的边处理逻辑）
        graph_edges = self._extract_graph_features_twibot(edge_df, user_data)
        
        # 7. 处理标签和用户ID
        user_ids = list(user_data['id'])
        labels = self._process_labels_twibot(label_df, user_ids)
        
        return PlatformData(
            user_ids=user_ids,
            user_texts=user_texts,
            labels=labels,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            graph_edges=graph_edges,
            platform_name="Twibot-20",
            language="en",
            feature_description={
                "numerical": ["followers_count", "following_count", "listed_count", 
                             "username_length", "account_age_days"],
                "categorical": ["is_verified", "is_protected", "has_default_avatar"],
                "text": ["description", "aggregated_tweets"]
            }
        )
    
    def process_misbot(self) -> PlatformData:
        """处理Misbot数据集（适配到统一格式）"""
        print("Processing Misbot dataset...")
        
        # 1. 加载Misbot数据
        node_df = pd.read_json("./dataset/Misbot/node.json")
        edge_df = pd.read_json("./dataset/Misbot/edge.json")
        label_df = pd.read_json("./dataset/Misbot/label.json")
        
        # 2. 分离用户和微博
        user_data, tweet_data = self._split_user_tweet_misbot(node_df)
        
        # 3. 提取文本特征
        user_texts = self._extract_text_features_misbot(user_data, tweet_data, edge_df)
        
        # 4. 提取数值特征（适配Misbot的numerical字段）
        numerical_features = self._extract_numerical_features_misbot(user_data)
        
        # 5. 提取分类特征（从categorical字段推断）
        categorical_features = self._extract_categorical_features_misbot(user_data)
        
        # 6. 提取图结构
        graph_edges = self._extract_graph_features_misbot(edge_df, user_data)
        
        # 7. 处理标签和用户ID
        user_ids = list(user_data['ID'])
        labels = self._process_labels_misbot(label_df, user_ids)
        
        return PlatformData(
            user_ids=user_ids,
            user_texts=user_texts,
            labels=labels,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            graph_edges=graph_edges,
            platform_name="Misbot",
            language="zh",
            feature_description={
                "numerical": ["followers_count", "following_count", "post_count", 
                             "username_length", "account_age_days"],
                "categorical": ["is_verified", "is_vip", "has_avatar"],
                "text": ["description", "aggregated_posts"]
            }
        )
    
    def _split_user_tweet_twibot(self, node_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分离Twibot-20的用户和推文数据"""
        # 转置DataFrame以便按ID前缀分离
        node_dict = node_df.to_dict()
        
        users = {}
        tweets = {}
        
        for node_id, node_data in node_dict.items():
            if node_id.startswith('u'):
                users[node_id] = node_data
            elif node_id.startswith('t'):
                tweets[node_id] = node_data
        
        user_df = pd.DataFrame.from_dict(users, orient='index').reset_index()
        user_df.rename(columns={'index': 'id'}, inplace=True)
        
        tweet_df = pd.DataFrame.from_dict(tweets, orient='index').reset_index()
        tweet_df.rename(columns={'index': 'id'}, inplace=True)
        
        return user_df, tweet_df
    
    def _split_user_tweet_misbot(self, node_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分离Misbot的用户和微博数据"""
        node_dict = node_df.to_dict()
        
        users = {}
        tweets = {}
        
        for node_id, node_data in node_dict.items():
            if isinstance(node_data, dict):
                if node_data.get('type') == 'user':
                    users[node_id] = node_data
                elif node_data.get('type') == 'tweet':
                    tweets[node_id] = node_data
        
        user_df = pd.DataFrame.from_dict(users, orient='index').reset_index()
        user_df.rename(columns={'index': 'ID'}, inplace=True)
        
        tweet_df = pd.DataFrame.from_dict(tweets, orient='index').reset_index()
        tweet_df.rename(columns={'index': 'ID'}, inplace=True)
        
        return user_df, tweet_df
    
    def _extract_text_features_twibot(self, user_df: pd.DataFrame, 
                                    tweet_df: pd.DataFrame, 
                                    edge_df: pd.DataFrame) -> List[str]:
        """提取Twibot-20文本特征（借鉴原作者方法）"""
        print("Extracting text features for Twibot-20...")
        
        # 1. 构建用户-推文映射（借鉴原作者逻辑）
        user_tweets_map = self._build_user_tweets_mapping_twibot(user_df, tweet_df, edge_df)
        
        # 2. 聚合每个用户的文本
        user_texts = []
        for _, user_row in tqdm(user_df.iterrows(), total=len(user_df)):
            user_id = user_row['id']
            
            # 用户描述
            description = user_row.get('description', '') or ''
            
            # 用户推文（最多20条，借鉴原作者限制）
            user_tweet_ids = user_tweets_map.get(user_id, [])
            tweet_texts = []
            
            for tweet_id in user_tweet_ids[:20]:  # 限制推文数量
                tweet_row = tweet_df[tweet_df['id'] == tweet_id]
                if not tweet_row.empty:
                    tweet_text = tweet_row.iloc[0].get('text', '') or ''
                    if tweet_text.strip():
                        tweet_texts.append(tweet_text)
            
            # 拼接描述和推文
            combined_text = description
            if tweet_texts:
                combined_text += " " + " ".join(tweet_texts)
            
            user_texts.append(combined_text.strip())
        
        return user_texts
    
    def _extract_text_features_misbot(self, user_df: pd.DataFrame,
                                    tweet_df: pd.DataFrame,
                                    edge_df: pd.DataFrame) -> List[str]:
        """提取Misbot文本特征"""
        print("Extracting text features for Misbot...")
        
        # 构建用户-微博映射
        user_tweets_map = self._build_user_tweets_mapping_misbot(user_df, tweet_df, edge_df)
        
        user_texts = []
        for _, user_row in tqdm(user_df.iterrows(), total=len(user_df)):
            user_id = user_row['ID']
            
            # 用户描述
            profile = user_row.get('profile', {})
            if isinstance(profile, dict):
                description = profile.get('description', '') or ''
            else:
                description = ''
            
            # 用户微博
            user_tweet_ids = user_tweets_map.get(user_id, [])
            tweet_texts = []
            
            for tweet_id in user_tweet_ids[:20]:  # 同样限制20条
                tweet_row = tweet_df[tweet_df['ID'] == tweet_id]
                if not tweet_row.empty:
                    tweet_text = tweet_row.iloc[0].get('content', '') or ''
                    if tweet_text.strip():
                        tweet_texts.append(tweet_text)
            
            # 拼接描述和微博
            combined_text = description
            if tweet_texts:
                combined_text += " " + " ".join(tweet_texts)
            
            user_texts.append(combined_text.strip())
        
        return user_texts
    
    def _extract_numerical_features_twibot(self, user_df: pd.DataFrame) -> np.ndarray:
        """提取Twibot-20数值特征（借鉴原作者的5维特征）"""
        print("Extracting numerical features for Twibot-20...")
        
        features = []
        
        for _, user_row in tqdm(user_df.iterrows(), total=len(user_df)):
            user_features = []
            
            # 1. followers_count
            public_metrics = user_row.get('public_metrics')
            if isinstance(public_metrics, dict):
                followers_count = public_metrics.get('followers_count', 0) or 0
                following_count = public_metrics.get('following_count', 0) or 0
                listed_count = public_metrics.get('listed_count', 0) or 0
            else:
                followers_count = following_count = listed_count = 0
            
            user_features.extend([followers_count, following_count, listed_count])
            
            # 2. username_length
            username = user_row.get('username', '') or ''
            username_length = len(username)
            user_features.append(username_length)
            
            # 3. account_age_days（借鉴原作者计算方法）
            created_at = user_row.get('created_at', '')
            if created_at:
                try:
                    # 解析创建时间
                    created_date = pd.to_datetime(created_at)
                    reference_date = pd.to_datetime('2020-09-01')  # 借鉴原作者的参考日期
                    age_days = (reference_date - created_date).days
                    age_days = max(0, age_days)  # 确保非负
                except:
                    age_days = 0
            else:
                age_days = 0
            
            user_features.append(age_days)
            
            features.append(user_features)
        
        # 转换为numpy数组并标准化（借鉴原作者方法）
        features_array = np.array(features, dtype=np.float32)
        
        # Z-score标准化
        for i in range(features_array.shape[1]):
            col = features_array[:, i]
            mean_val = np.mean(col)
            std_val = np.std(col)
            if std_val > 0:
                features_array[:, i] = (col - mean_val) / std_val
            else:
                features_array[:, i] = 0
        
        return features_array
    
    def _extract_numerical_features_misbot(self, user_df: pd.DataFrame) -> np.ndarray:
        """提取Misbot数值特征（适配到5维）"""
        print("Extracting numerical features for Misbot...")
        
        features = []
        
        for _, user_row in tqdm(user_df.iterrows(), total=len(user_df)):
            user_features = []
            
            # 从profile.numerical提取原始3维特征
            profile = user_row.get('profile', {})
            if isinstance(profile, dict):
                numerical = profile.get('numerical', [0, 0, 0])
                if isinstance(numerical, list) and len(numerical) >= 3:
                    # 粉丝数、关注数、微博数
                    user_features.extend(numerical[:3])
                else:
                    user_features.extend([0, 0, 0])
            else:
                user_features.extend([0, 0, 0])
            
            # 用户名长度（从ID推断）
            user_id = user_row.get('ID', '')
            # 简单估算：去掉前缀后的长度
            if user_id.startswith('train_u'):
                username_length = len(user_id) - 7  # 去掉'train_u'前缀
            else:
                username_length = len(user_id)
            user_features.append(username_length)
            
            # 账户年龄（设为固定值，因为Misbot没有时间信息）
            user_features.append(1000)  # 假设账户年龄
            
            features.append(user_features)
        
        # 标准化处理
        features_array = np.array(features, dtype=np.float32)
        
        for i in range(features_array.shape[1]):
            col = features_array[:, i]
            mean_val = np.mean(col)
            std_val = np.std(col)
            if std_val > 0:
                features_array[:, i] = (col - mean_val) / std_val
            else:
                features_array[:, i] = 0
        
        return features_array
    
    def _extract_categorical_features_twibot(self, user_df: pd.DataFrame) -> np.ndarray:
        """提取Twibot-20分类特征（借鉴原作者方法）"""
        print("Extracting categorical features for Twibot-20...")
        
        features = []
        
        for _, user_row in tqdm(user_df.iterrows(), total=len(user_df)):
            user_features = []
            
            # 1. is_verified
            verified = user_row.get('verified', False)
            if isinstance(verified, str):
                verified = verified.strip().lower() == 'true'
            user_features.append(1 if verified else 0)
            
            # 2. is_protected
            protected = user_row.get('protected', False)
            if isinstance(protected, str):
                protected = protected.strip().lower() == 'true'
            user_features.append(1 if protected else 0)
            
            # 3. has_default_avatar（借鉴原作者逻辑）
            profile_image_url = user_row.get('profile_image_url', '')
            has_default = (profile_image_url == 'http://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png' 
                          or not profile_image_url)
            user_features.append(1 if has_default else 0)
            
            features.append(user_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_categorical_features_misbot(self, user_df: pd.DataFrame) -> np.ndarray:
        """提取Misbot分类特征（从categorical字段推断）"""
        print("Extracting categorical features for Misbot...")
        
        features = []
        
        for _, user_row in tqdm(user_df.iterrows(), total=len(user_df)):
            user_features = []
            
            profile = user_row.get('profile', {})
            if isinstance(profile, dict):
                categorical = profile.get('categorical', [0] * 20)
                if isinstance(categorical, list) and len(categorical) >= 20:
                    # 从20维categorical中提取关键特征
                    # 认证状态（前2维）
                    is_verified = 1 if categorical[0] == 1 else 0
                    user_features.append(is_verified)
                    
                    # VIP状态（第3-4维）
                    is_vip = 1 if categorical[2] == 1 else 0
                    user_features.append(is_vip)
                    
                    # 假设有头像（简单推断）
                    has_avatar = 1  # 默认有头像
                    user_features.append(has_avatar)
                else:
                    user_features.extend([0, 0, 1])  # 默认值
            else:
                user_features.extend([0, 0, 1])  # 默认值
            
            features.append(user_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_graph_features_twibot(self, edge_df: pd.DataFrame, 
                                     user_df: pd.DataFrame) -> List[Tuple]:
        """提取Twibot-20图特征（借鉴原作者边处理逻辑）"""
        print("Extracting graph features for Twibot-20...")
        
        # 建立用户ID到索引的映射
        user_ids = list(user_df['id'])
        uid_to_index = {uid: i for i, uid in enumerate(user_ids)}
        
        edges = []
        
        # 处理边关系（借鉴原作者逻辑）
        edge_dict = edge_df.to_dict()
        
        for source_id, relations in tqdm(edge_dict.items()):
            if source_id not in uid_to_index:
                continue
                
            source_idx = uid_to_index[source_id]
            
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, list) and len(relation) >= 2:
                        relation_type, target_id = relation[0], relation[1]
                        
                        # 只处理用户间关系（跳过post关系）
                        if relation_type in ['friend', 'follow'] and target_id in uid_to_index:
                            target_idx = uid_to_index[target_id]
                            edge_type = 0 if relation_type == 'friend' else 1
                            edges.append((source_idx, target_idx, edge_type))
        
        return edges
    
    def _extract_graph_features_misbot(self, edge_df: pd.DataFrame,
                                     user_df: pd.DataFrame) -> List[Tuple]:
        """提取Misbot图特征"""
        print("Extracting graph features for Misbot...")
        
        # 建立用户ID到索引的映射
        user_ids = list(user_df['ID'])
        uid_to_index = {uid: i for i, uid in enumerate(user_ids)}
        
        edges = []
        
        # 处理边关系
        edge_dict = edge_df.to_dict()
        
        for source_id, relations in tqdm(edge_dict.items()):
            if source_id not in uid_to_index:
                continue
                
            source_idx = uid_to_index[source_id]
            
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, list) and len(relation) >= 2:
                        relation_type, target_id = relation[0], relation[1]
                        
                        # 只处理用户间关系
                        if relation_type in ['mention', 'retweet', 'follow'] and target_id in uid_to_index:
                            target_idx = uid_to_index[target_id]
                            # 映射关系类型
                            if relation_type == 'follow':
                                edge_type = 1  # 与Twibot保持一致
                            else:
                                edge_type = 2  # mention/retweet
                            edges.append((source_idx, target_idx, edge_type))
        
        return edges
    
    def _build_user_tweets_mapping_twibot(self, user_df: pd.DataFrame,
                                        tweet_df: pd.DataFrame,
                                        edge_df: pd.DataFrame) -> Dict[str, List[str]]:
        """构建Twibot-20用户-推文映射（借鉴原作者方法）"""
        user_tweets = {uid: [] for uid in user_df['id']}
        
        edge_dict = edge_df.to_dict()
        
        for source_id, relations in edge_dict.items():
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, list) and len(relation) >= 2:
                        relation_type, target_id = relation[0], relation[1]
                        
                        if relation_type == 'post' and source_id in user_tweets:
                            user_tweets[source_id].append(target_id)
        
        return user_tweets
    
    def _build_user_tweets_mapping_misbot(self, user_df: pd.DataFrame,
                                        tweet_df: pd.DataFrame,
                                        edge_df: pd.DataFrame) -> Dict[str, List[str]]:
        """构建Misbot用户-微博映射"""
        user_tweets = {uid: [] for uid in user_df['ID']}
        
        edge_dict = edge_df.to_dict()
        
        for source_id, relations in edge_dict.items():
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, list) and len(relation) >= 2:
                        relation_type, target_id = relation[0], relation[1]
                        
                        if relation_type == 'post' and source_id in user_tweets:
                            user_tweets[source_id].append(target_id)
        
        return user_tweets
    
    def _process_labels_twibot(self, label_df: pd.DataFrame, user_ids: List[str]) -> Dict[str, int]:
        """处理Twibot-20标签"""
        labels = {}
        label_dict = label_df.to_dict()
        
        for user_id in user_ids:
            if user_id in label_dict:
                label_str = label_dict[user_id]
                labels[user_id] = 0 if label_str == 'human' else 1
            else:
                labels[user_id] = 0  # 默认为human
        
        return labels
    
    def _process_labels_misbot(self, label_df: pd.DataFrame, user_ids: List[str]) -> Dict[str, int]:
        """处理Misbot标签"""
        labels = {}
        label_dict = label_df.to_dict()
        
        for user_id in user_ids:
            if user_id in label_dict:
                label_str = label_dict[user_id]
                labels[user_id] = 0 if label_str == 'human' else 1
            else:
                labels[user_id] = 0  # 默认为human
        
        return labels
    
    def save_processed_data(self, platform_data: PlatformData, 
                          output_prefix: str = None) -> Dict[str, str]:
        """保存处理后的数据"""
        if output_prefix is None:
            output_prefix = platform_data.platform_name.lower()
        
        saved_files = {}
        
        # 保存文本特征（需要编码）
        print(f"Encoding text features for {platform_data.platform_name}...")
        text_embeddings = self._encode_texts(platform_data.user_texts)
        text_path = self.output_dir / f"{output_prefix}_text_features.pt"
        torch.save(text_embeddings, text_path)
        saved_files['text_features'] = str(text_path)
        
        # 保存数值特征
        if platform_data.numerical_features is not None:
            num_path = self.output_dir / f"{output_prefix}_numerical_features.pt"
            torch.save(torch.tensor(platform_data.numerical_features), num_path)
            saved_files['numerical_features'] = str(num_path)
        
        # 保存分类特征
        if platform_data.categorical_features is not None:
            cat_path = self.output_dir / f"{output_prefix}_categorical_features.pt"
            torch.save(torch.tensor(platform_data.categorical_features), cat_path)
            saved_files['categorical_features'] = str(cat_path)
        
        # 保存图结构
        if platform_data.graph_edges is not None:
            edge_indices = []
            edge_types = []
            for edge in platform_data.graph_edges:
                edge_indices.append([edge[0], edge[1]])
                edge_types.append(edge[2])
            
            if edge_indices:
                edge_index_path = self.output_dir / f"{output_prefix}_edge_index.pt"
                edge_type_path = self.output_dir / f"{output_prefix}_edge_types.pt"
                
                torch.save(torch.tensor(edge_indices).t(), edge_index_path)
                torch.save(torch.tensor(edge_types), edge_type_path)
                
                saved_files['edge_index'] = str(edge_index_path)
                saved_files['edge_types'] = str(edge_type_path)
        
        # 保存标签
        label_path = self.output_dir / f"{output_prefix}_labels.pt"
        label_tensor = torch.tensor([platform_data.labels.get(uid, 0) 
                                   for uid in platform_data.user_ids])
        torch.save(label_tensor, label_path)
        saved_files['labels'] = str(label_path)
        
        # 保存用户ID映射
        uid_path = self.output_dir / f"{output_prefix}_user_ids.json"
        with open(uid_path, 'w') as f:
            json.dump(platform_data.user_ids, f)
        saved_files['user_ids'] = str(uid_path)
        
        # 保存元信息
        meta_path = self.output_dir / f"{output_prefix}_metadata.json"
        metadata = {
            'platform_name': platform_data.platform_name,
            'language': platform_data.language,
            'feature_description': platform_data.feature_description,
            'num_users': len(platform_data.user_ids),
            'saved_files': saved_files
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {platform_data.platform_name} data to {len(saved_files)} files")
        return saved_files
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """编码文本为向量（借鉴原作者方法但支持多语言）"""
        embeddings = []
        
        for text in tqdm(texts, desc="Encoding texts"):
            if not text or text.strip() == '':
                # 空文本用零向量
                embeddings.append(torch.zeros(self.text_dim))
            else:
                try:
                    # 使用transformers pipeline编码
                    features = self.text_encoder(text)
                    
                    # 平均池化（借鉴原作者方法）
                    if isinstance(features, list) and len(features) > 0:
                        feature_tensor = torch.tensor(features[0])
                        # 对所有token求平均
                        pooled_feature = feature_tensor.mean(dim=0)
                        embeddings.append(pooled_feature)
                    else:
                        embeddings.append(torch.zeros(self.text_dim))
                except Exception as e:
                    print(f"Error encoding text: {e}")
                    embeddings.append(torch.zeros(self.text_dim))
        
        return torch.stack(embeddings)

# 使用示例和配置
def main():
    """主函数：处理两个数据集"""
    
    config = {
        'output_dir': 'processed_data',
        'text_model': 'xlm-roberta-base',  # 多语言支持
        'max_text_length': 128,
        'text_dim': 768,
        'max_numerical_features': 5,
        'max_categorical_features': 3
    }
    
    preprocessor = UnifiedPreprocessor(config)
    
    # 处理Twibot-20
    print("=" * 50)
    print("Processing Twibot-20...")
    twibot_data = preprocessor.process_twibot20()
    twibot_files = preprocessor.save_processed_data(twibot_data, 'twibot20')
    
    # 处理Misbot
    print("=" * 50)
    print("Processing Misbot...")
    misbot_data = preprocessor.process_misbot()
    misbot_files = preprocessor.save_processed_data(misbot_data, 'misbot')
    
    print("=" * 50)
    print("Processing completed!")
    print(f"Twibot-20: {len(twibot_data.user_ids)} users")
    print(f"Misbot: {len(misbot_data.user_ids)} users")

if __name__ == "__main__":
    main()