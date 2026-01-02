#!/usr/bin/env python
"""
预计算文本嵌入

在训练前一次性计算所有用户的文本嵌入并保存为 .pt 文件。
训练时直接加载预计算的嵌入，避免重复的 Transformer 推理。

使用方法:
    python precompute_text_embeddings.py --dataset twibot20
    python precompute_text_embeddings.py --dataset misbot
    python precompute_text_embeddings.py --dataset all

输出:
    processed_data/{dataset}/text_embeddings.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_user_texts(data_path: Path) -> Dict[int, dict]:
    """加载用户文本数据"""
    texts_path = data_path / "user_texts.json"
    if not texts_path.exists():
        raise FileNotFoundError(f"user_texts.json not found in {data_path}")
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        raw_texts = json.load(f)
    
    # 转换键为整数
    return {int(k): v for k, v in raw_texts.items()}


def combine_text_fields(text_data) -> str:
    """将用户文本字段合并为单个字符串"""
    if isinstance(text_data, dict):
        description = text_data.get('description', '') or ''
        tweets = text_data.get('tweets', []) or []
        tweets_text = ' '.join(tweets) if tweets else ''
        combined = f"{description} {tweets_text}".strip()
        return combined if combined else ""
    elif isinstance(text_data, str):
        return text_data if text_data else ""
    else:
        return ""


def precompute_embeddings(
    data_path: Path,
    model_name: str = "xlm-roberta-base",
    output_dim: int = 256,
    max_length: int = 128,
    batch_size: int = 32,
    device: str = "cuda"
) -> torch.Tensor:
    """
    预计算所有用户的文本嵌入
    
    Args:
        data_path: 数据集路径
        model_name: 预训练模型名称
        output_dim: 输出嵌入维度
        max_length: 最大token长度
        batch_size: 批处理大小
        device: 计算设备
    
    Returns:
        Tensor[num_users, output_dim] 所有用户的文本嵌入
    """
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # 加载用户文本
    logger.info("Loading user texts...")
    user_texts = load_user_texts(data_path)
    
    # 获取用户数量（从 labels.pt）
    labels_path = data_path / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.pt not found in {data_path}")
    labels = torch.load(labels_path, weights_only=True)
    num_users = len(labels)
    logger.info(f"Total users: {num_users}")
    logger.info(f"Users with text: {len(user_texts)}")
    
    # 加载模型和分词器
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # 投影层
    hidden_size = model.config.hidden_size
    projection = torch.nn.Linear(hidden_size, output_dim).to(device)
    
    # 初始化嵌入张量
    embeddings = torch.zeros(num_users, output_dim)
    
    # 准备所有文本
    all_texts = []
    all_indices = []
    for idx in range(num_users):
        text_data = user_texts.get(idx, "")
        text = combine_text_fields(text_data)
        all_texts.append(text)
        all_indices.append(idx)
    
    # 批量处理
    logger.info(f"Computing embeddings (batch_size={batch_size}, max_length={max_length})...")
    
    with torch.no_grad():
        for i in tqdm(range(0, num_users, batch_size), desc="Processing"):
            batch_texts = all_texts[i:i+batch_size]
            batch_indices = all_indices[i:i+batch_size]
            
            # 跳过全空批次
            non_empty = [t for t in batch_texts if t.strip()]
            if not non_empty:
                continue
            
            # Tokenize
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            
            # Project
            batch_embeddings = projection(cls_output)
            
            # 将空文本的嵌入置零
            for j, text in enumerate(batch_texts):
                if not text.strip():
                    batch_embeddings[j] = 0
            
            # 存储
            embeddings[batch_indices] = batch_embeddings.cpu()
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Precompute text embeddings")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        choices=["twibot20", "misbot", "all"],
        help="Dataset to process (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xlm-roberta-base",
        help="Pretrained model name (default: xlm-roberta-base)"
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
        help="Output embedding dimension (default: 256)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length (default: 128)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="processed_data",
        help="Data directory (default: processed_data)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # 确定要处理的数据集
    if args.dataset == "all":
        datasets = ["twibot20", "misbot"]
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        data_path = data_dir / dataset
        
        if not data_path.exists():
            logger.warning(f"Dataset directory not found: {data_path}, skipping")
            continue
        
        # 检查是否有 user_texts.json
        if not (data_path / "user_texts.json").exists():
            logger.warning(f"user_texts.json not found in {data_path}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset}")
        logger.info(f"{'='*60}")
        
        try:
            embeddings = precompute_embeddings(
                data_path=data_path,
                model_name=args.model,
                output_dim=args.output_dim,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device
            )
            
            # 保存嵌入
            output_path = data_path / "text_embeddings.pt"
            torch.save(embeddings, output_path)
            logger.info(f"Saved embeddings to: {output_path}")
            logger.info(f"Shape: {embeddings.shape}")
            
            # 保存元数据
            meta_path = data_path / "text_embeddings_meta.json"
            meta = {
                "model_name": args.model,
                "output_dim": args.output_dim,
                "max_length": args.max_length,
                "num_users": embeddings.shape[0]
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Saved metadata to: {meta_path}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
