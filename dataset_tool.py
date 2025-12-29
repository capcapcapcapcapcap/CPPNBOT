from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
from torch_geometric.data import Data, HeteroData

# 简化的数据目录配置
DATASET_DIR = Path("./dataset")

# 当前项目支持的数据集
SUPPORTED_DATASETS = ["Twibot-20", "Misbot"]

def split_user_and_tweet(df):
    """分离用户和推文数据"""
    df = df[df.id.str.len() > 0]
    return df[df.id.str.contains("^u")], df[df.id.str.contains("^t")]

def fast_merge(dataset="Twibot-20"):
    """加载并合并Twibot-20数据集"""
    assert dataset in SUPPORTED_DATASETS, f"Unsupported dataset {dataset}. Supported: {SUPPORTED_DATASETS}"
    
    dataset_dir = DATASET_DIR / dataset
    node_info = pd.read_json(dataset_dir / "node.json")
    label = pd.read_json(dataset_dir / "label.json")
    split = pd.read_json(dataset_dir / "split.json")
    
    user, tweet = split_user_and_tweet(node_info)
    
    # 构建标签和分割映射
    id_to_label = {row["id"]: row["label"] for _, row in label.iterrows()}
    id_to_split = {row["id"]: row["split"] for _, row in split.iterrows()}
    
    # 为用户添加标签和分割信息
    user = user.copy()
    user["label"] = user["id"].map(id_to_label).fillna("None")
    user["split"] = user["id"].map(id_to_split).fillna("None")
        
    return user, tweet

@torch.no_grad()
def extract_text_features(texts, model_name='roberta-base'):
    """提取文本特征"""
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    
    feats = []
    for text in tqdm(texts, desc="Extracting text features"):
        if text is None or str(text).strip() == "":
            feats.append(torch.zeros(768))
        else:
            encoded_input = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=512)
            feats.append(model(**encoded_input)["pooler_output"][0])
        
    return torch.stack(feats, dim=0)

def build_graph_from_edges(dataset="Twibot-20"):
    """从边文件构建图结构"""
    assert dataset in SUPPORTED_DATASETS, f"Unsupported dataset {dataset}"
    
    dataset_dir = DATASET_DIR / dataset
    user, tweet = fast_merge(dataset)
    
    # 构建用户索引映射
    user_ids = list(user.id)
    uid_to_index = {uid: i for i, uid in enumerate(user_ids)}
    
    # 读取边信息
    edge_df = pd.read_json(dataset_dir / "edge.json")
    
    # 过滤用户间的关系（排除post关系）
    user_edges = edge_df[edge_df["relation"] != "post"]
    
    # 构建边索引
    edge_list = []
    for _, row in user_edges.iterrows():
        src, dst = row["source_id"], row["target_id"]
        if src in uid_to_index and dst in uid_to_index:
            edge_list.append([uid_to_index[src], uid_to_index[dst]])
    
    if len(edge_list) > 0:
        edge_index = torch.LongTensor(edge_list).t()
    else:
        edge_index = torch.LongTensor([[], []])
    
    # 构建标签
    labels = [0 if label == "human" else 1 for label in user.label]
    labels = torch.LongTensor(labels)
    
    return user, edge_index, labels, uid_to_index
def df_to_mask(uid_with_label, uid_to_user_index, phase="train"):
    """将用户ID转换为掩码索引"""
    user_list = list(uid_with_label[uid_with_label.split == phase].id)
    phase_index = [uid_to_user_index[uid] for uid in user_list if uid in uid_to_user_index]
    return torch.LongTensor(phase_index)

def load_misbot_data():
    """加载Misbot数据集"""
    dataset_dir = DATASET_DIR / "Misbot"
    
    # 读取训练数据
    train_data = []
    with open(dataset_dir / "train_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(pd.read_json(line, typ='series'))
    
    train_df = pd.DataFrame(train_data)
    
    # 读取推理数据（可选）
    inference_data = []
    inference_file = dataset_dir / "inference_data.jsonl"
    if inference_file.exists():
        with open(inference_file, 'r', encoding='utf-8') as f:
            for line in f:
                inference_data.append(pd.read_json(line, typ='series'))
        inference_df = pd.DataFrame(inference_data)
    else:
        inference_df = pd.DataFrame()
    
    return train_df, inference_df

# 简化的测试函数
if __name__ == "__main__":
    print("Testing Twibot-20 data loading...")
    try:
        user, tweet = fast_merge("Twibot-20")
        print(f"✓ Loaded {len(user)} users and {len(tweet)} tweets")
        
        user_data, edge_index, labels, uid_to_index = build_graph_from_edges("Twibot-20")
        print(f"✓ Built graph with {len(user_data)} nodes and {edge_index.shape[1]} edges")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        
    print("\nTesting Misbot data loading...")
    try:
        train_df, inference_df = load_misbot_data()
        print(f"✓ Loaded {len(train_df)} training samples and {len(inference_df)} inference samples")
    except Exception as e:
        print(f"✗ Error: {e}")