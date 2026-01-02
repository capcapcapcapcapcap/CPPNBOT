#!/usr/bin/env python
"""
预计算图嵌入

在训练前一次性计算所有用户的图嵌入并保存为 .pt 文件。
训练时直接加载预计算的嵌入，避免重复的 RGCN 推理。

使用方法:
    python precompute_graph_embeddings.py --dataset twibot20
    python precompute_graph_embeddings.py --dataset misbot
    python precompute_graph_embeddings.py --dataset all

输出:
    processed_data/{dataset}/graph_embeddings.pt
"""

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def precompute_embeddings(
    data_path: Path,
    num_input_dim: int = 8,
    num_hidden_dim: int = 32,
    num_output_dim: int = 64,
    cat_num_categories: list = None,
    cat_embedding_dim: int = 16,
    cat_output_dim: int = 32,
    text_embeddings_path: Optional[Path] = None,
    graph_input_dim: int = 96,
    graph_hidden_dim: int = 128,
    graph_output_dim: int = 128,
    graph_num_relations: int = 2,
    graph_num_layers: int = 2,
    graph_dropout: float = 0.0,  # 推理时不用dropout
    graph_num_bases: Optional[int] = None,
    device: str = "cuda",
    use_fp16: bool = True
) -> torch.Tensor:
    """
    预计算所有用户的图嵌入
    
    注意: RGCN 需要完整图结构进行消息传递，无法分 batch 处理。
    对于大图，主要优化是使用 FP16 和确保数据在 GPU 上。
    
    Args:
        data_path: 数据集路径
        num_input_dim: 数值特征输入维度
        num_hidden_dim: 数值编码器隐藏层维度
        num_output_dim: 数值编码器输出维度
        cat_num_categories: 分类特征类别数列表
        cat_embedding_dim: 分类嵌入维度
        cat_output_dim: 分类编码器输出维度
        text_embeddings_path: 预计算文本嵌入路径（可选）
        graph_input_dim: 图编码器输入维度
        graph_hidden_dim: 图编码器隐藏层维度
        graph_output_dim: 图编码器输出维度
        graph_num_relations: 关系类型数量
        graph_num_layers: RGCN层数
        graph_dropout: Dropout比率 (推理时建议设为0)
        graph_num_bases: 基分解数量
        device: 计算设备
        use_fp16: 是否使用半精度加速 (仅GPU)
    
    Returns:
        Tensor[num_users, graph_output_dim] 所有用户的图嵌入
    """
    if cat_num_categories is None:
        cat_num_categories = [2, 2, 2, 2, 2]
    
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
        use_fp16 = False
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"Using FP16: {use_fp16}")
    
    # 加载数据
    logger.info("Loading data...")
    num_features = torch.load(data_path / "num_features.pt", weights_only=True)
    cat_features = torch.load(data_path / "cat_features.pt", weights_only=True)
    edge_index = torch.load(data_path / "edge_index.pt", weights_only=True)
    edge_type = torch.load(data_path / "edge_type.pt", weights_only=True)
    
    num_users = num_features.size(0)
    num_edges = edge_index.size(1)
    logger.info(f"Total users: {num_users:,}")
    logger.info(f"Total edges: {num_edges:,}")
    
    # 加载预计算的文本嵌入（如果有）
    text_embeddings = None
    if text_embeddings_path and text_embeddings_path.exists():
        text_embeddings = torch.load(text_embeddings_path, weights_only=True)
        logger.info(f"Loaded text embeddings: {text_embeddings.shape}")
    elif (data_path / "text_embeddings.pt").exists():
        text_embeddings = torch.load(data_path / "text_embeddings.pt", weights_only=True)
        logger.info(f"Loaded text embeddings: {text_embeddings.shape}")
    
    # 创建编码器
    logger.info("Creating encoders...")
    from src.models.encoders.numerical import NumericalEncoder
    from src.models.encoders.categorical import CategoricalEncoder
    from src.models.encoders.graph import GraphEncoder
    
    num_encoder = NumericalEncoder(
        input_dim=num_input_dim,
        hidden_dim=num_hidden_dim,
        output_dim=num_output_dim
    ).to(device)
    
    cat_encoder = CategoricalEncoder(
        num_categories=cat_num_categories,
        embedding_dim=cat_embedding_dim,
        output_dim=cat_output_dim
    ).to(device)
    
    graph_encoder = GraphEncoder(
        input_dim=graph_input_dim,
        hidden_dim=graph_hidden_dim,
        output_dim=graph_output_dim,
        num_relations=graph_num_relations,
        num_layers=graph_num_layers,
        dropout=graph_dropout,
        num_bases=graph_num_bases
    ).to(device)
    
    # 设置为评估模式
    num_encoder.eval()
    cat_encoder.eval()
    graph_encoder.eval()
    
    # 移动数据到设备
    logger.info("Moving data to device...")
    num_features = num_features.to(device)
    cat_features = cat_features.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    if text_embeddings is not None:
        text_embeddings = text_embeddings.to(device)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU Memory allocated: {mem_allocated:.2f} GB")
    
    # 计算图嵌入
    logger.info("Computing graph embeddings...")
    
    with torch.no_grad():
        # 使用混合精度
        autocast_ctx = torch.cuda.amp.autocast() if (use_fp16 and device.type == "cuda") else nullcontext()
        
        with autocast_ctx:
            # 编码数值和分类特征
            num_embed = num_encoder(num_features)
            cat_embed = cat_encoder(cat_features)
            
            # 拼接作为图编码器输入
            if text_embeddings is not None:
                graph_input = torch.cat([num_embed, cat_embed, text_embeddings], dim=1)
            else:
                graph_input = torch.cat([num_embed, cat_embed], dim=1)
            
            logger.info(f"Graph input shape: {graph_input.shape}")
            
            # 图编码 (RGCN 需要完整图，无法分batch)
            embeddings = graph_encoder(graph_input.float(), edge_index, edge_type)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    return embeddings.float().cpu()


def main():
    parser = argparse.ArgumentParser(description="Precompute graph embeddings")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        choices=["twibot20", "misbot", "all"],
        help="Dataset to process (default: all)"
    )
    parser.add_argument(
        "--graph-hidden-dim",
        type=int,
        default=128,
        help="Graph encoder hidden dimension (default: 128)"
    )
    parser.add_argument(
        "--graph-output-dim",
        type=int,
        default=128,
        help="Graph encoder output dimension (default: 128)"
    )
    parser.add_argument(
        "--graph-num-relations",
        type=int,
        default=2,
        help="Number of relation types (default: 2)"
    )
    parser.add_argument(
        "--graph-num-layers",
        type=int,
        default=2,
        help="Number of RGCN layers (default: 2)"
    )
    parser.add_argument(
        "--use-text",
        action="store_true",
        help="Include text embeddings in graph input"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 mixed precision (default: enabled on GPU)"
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
        
        # 检查必需文件
        required_files = ["num_features.pt", "cat_features.pt", "edge_index.pt", "edge_type.pt"]
        missing = [f for f in required_files if not (data_path / f).exists()]
        if missing:
            logger.warning(f"Missing files in {data_path}: {missing}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset}")
        logger.info(f"{'='*60}")
        
        # 计算图输入维度
        # num(64) + cat(32) + text(256, optional)
        graph_input_dim = 64 + 32
        if args.use_text and (data_path / "text_embeddings.pt").exists():
            graph_input_dim += 256
            logger.info("Including text embeddings in graph input")
        
        try:
            import time
            start_time = time.time()
            
            embeddings = precompute_embeddings(
                data_path=data_path,
                graph_input_dim=graph_input_dim,
                graph_hidden_dim=args.graph_hidden_dim,
                graph_output_dim=args.graph_output_dim,
                graph_num_relations=args.graph_num_relations,
                graph_num_layers=args.graph_num_layers,
                device=args.device,
                use_fp16=not args.no_fp16
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Computation time: {elapsed:.2f}s")
            
            # 保存嵌入
            output_path = data_path / "graph_embeddings.pt"
            torch.save(embeddings, output_path)
            logger.info(f"Saved embeddings to: {output_path}")
            logger.info(f"Shape: {embeddings.shape}")
            
            # 保存元数据
            meta_path = data_path / "graph_embeddings_meta.json"
            meta = {
                "graph_input_dim": graph_input_dim,
                "graph_hidden_dim": args.graph_hidden_dim,
                "graph_output_dim": args.graph_output_dim,
                "graph_num_relations": args.graph_num_relations,
                "graph_num_layers": args.graph_num_layers,
                "use_text": args.use_text,
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
