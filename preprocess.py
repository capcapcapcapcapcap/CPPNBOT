#!/usr/bin/env python
"""
统一预处理入口脚本

整合数据预处理、文本嵌入预计算和图嵌入预计算。

使用方法:
    # 完整预处理（数据 + 文本嵌入 + 图嵌入）
    python preprocess.py --dataset all --all
    
    # 只预处理数据
    python preprocess.py --dataset twibot20 --data
    
    # 只预计算文本嵌入
    python preprocess.py --dataset twibot20 --text
    
    # 只预计算图嵌入
    python preprocess.py --dataset twibot20 --graph
    
    # 组合使用
    python preprocess.py --dataset all --data --text
    python preprocess.py --dataset twibot20 --text --graph
    
    # 指定设备和批次大小
    python preprocess.py --dataset all --all --device cuda --batch-size 128
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_data(datasets: list, data_dir: str = "processed_data"):
    """预处理原始数据"""
    from preprocess_unified import UnifiedPreprocessor
    
    preprocessor = UnifiedPreprocessor(output_dir=data_dir)
    
    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Preprocessing data: {dataset}")
        logger.info(f"{'='*60}")
        try:
            preprocessor.preprocess(dataset)
        except Exception as e:
            logger.error(f"Error preprocessing {dataset}: {e}")
            import traceback
            traceback.print_exc()


def precompute_text_embeddings(
    datasets: list,
    data_dir: str = "processed_data",
    model_name: str = "xlm-roberta-base",
    max_length: int = 128,
    batch_size: int = 64,
    device: str = "cuda",
    use_fp16: bool = True
):
    """预计算文本嵌入（保存原始CLS输出，不做投影）"""
    import json
    import torch
    from contextlib import nullcontext
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer
    
    def load_user_texts(data_path: Path):
        texts_path = data_path / "user_texts.json"
        if not texts_path.exists():
            raise FileNotFoundError(f"user_texts.json not found in {data_path}")
        with open(texts_path, 'r', encoding='utf-8') as f:
            raw_texts = json.load(f)
        return {int(k): v for k, v in raw_texts.items()}
    
    def combine_text_fields(text_data):
        if isinstance(text_data, dict):
            description = text_data.get('description', '') or ''
            tweets = text_data.get('tweets', []) or []
            tweets_text = ' '.join(tweets) if tweets else ''
            combined = f"{description} {tweets_text}".strip()
            return combined if combined else ""
        elif isinstance(text_data, str):
            return text_data if text_data else ""
        return ""
    
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
        use_fp16 = False
    
    device_obj = torch.device(device)
    
    # 加载模型（只加载一次）
    logger.info(f"Loading text model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device_obj)
    model.eval()
    
    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")
    
    for dataset in datasets:
        data_path = Path(data_dir) / dataset
        
        if not data_path.exists():
            logger.warning(f"Dataset directory not found: {data_path}, skipping")
            continue
        
        if not (data_path / "user_texts.json").exists():
            logger.warning(f"user_texts.json not found in {data_path}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Computing text embeddings: {dataset}")
        logger.info(f"{'='*60}")
        
        try:
            # 加载数据
            user_texts = load_user_texts(data_path)
            labels = torch.load(data_path / "labels.pt", weights_only=True)
            num_users = len(labels)
            
            logger.info(f"Total users: {num_users:,}")
            logger.info(f"Device: {device_obj}, FP16: {use_fp16}")
            
            # 准备文本
            all_texts = [combine_text_fields(user_texts.get(idx, "")) for idx in range(num_users)]
            
            # 初始化嵌入（保存原始维度）
            embeddings = torch.zeros(num_users, hidden_size)
            
            # 批量处理
            with torch.no_grad():
                autocast_ctx = torch.amp.autocast('cuda') if (use_fp16 and device_obj.type == "cuda") else nullcontext()
                
                for i in tqdm(range(0, num_users, batch_size), desc="Processing"):
                    batch_texts = all_texts[i:i+batch_size]
                    batch_size_actual = len(batch_texts)
                    
                    encoding = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    
                    input_ids = encoding["input_ids"].to(device_obj)
                    attention_mask = encoding["attention_mask"].to(device_obj)
                    
                    with autocast_ctx:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        cls_output = outputs.last_hidden_state[:, 0, :].float()
                    
                    for j, text in enumerate(batch_texts):
                        if not text.strip():
                            cls_output[j] = 0
                    
                    embeddings[i:i+batch_size_actual] = cls_output.cpu()
            
            # 保存
            output_path = data_path / "text_embeddings.pt"
            torch.save(embeddings, output_path)
            logger.info(f"Saved: {output_path} ({embeddings.shape})")
            
            # 保存元数据
            meta = {
                "model_name": model_name,
                "hidden_size": hidden_size,
                "max_length": max_length,
                "num_users": num_users
            }
            with open(data_path / "text_embeddings_meta.json", 'w') as f:
                json.dump(meta, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()


def precompute_graph_embeddings(
    datasets: list,
    data_dir: str = "processed_data",
    graph_hidden_dim: int = 128,
    graph_output_dim: int = 128,
    graph_num_relations: int = 2,
    graph_num_layers: int = 2,
    device: str = "cuda",
    use_fp16: bool = True
):
    """预计算图嵌入
    
    图编码器输入固定为 num + cat 特征 (96维)，与文本模态独立。
    """
    import json
    import torch
    from contextlib import nullcontext
    
    from src.models.encoders.numerical import NumericalEncoder
    from src.models.encoders.categorical import CategoricalEncoder
    from src.models.encoders.graph import GraphEncoder
    
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
        use_fp16 = False
    
    device_obj = torch.device(device)
    
    for dataset in datasets:
        data_path = Path(data_dir) / dataset
        
        if not data_path.exists():
            logger.warning(f"Dataset directory not found: {data_path}, skipping")
            continue
        
        required_files = ["num_features.pt", "cat_features.pt", "edge_index.pt", "edge_type.pt"]
        missing = [f for f in required_files if not (data_path / f).exists()]
        if missing:
            logger.warning(f"Missing files in {data_path}: {missing}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Computing graph embeddings: {dataset}")
        logger.info(f"{'='*60}")
        
        try:
            import time
            start_time = time.time()
            
            # 加载数据
            num_features = torch.load(data_path / "num_features.pt", weights_only=True)
            cat_features = torch.load(data_path / "cat_features.pt", weights_only=True)
            edge_index = torch.load(data_path / "edge_index.pt", weights_only=True)
            edge_type = torch.load(data_path / "edge_type.pt", weights_only=True)
            
            num_users = num_features.size(0)
            logger.info(f"Total users: {num_users:,}")
            logger.info(f"Total edges: {edge_index.size(1):,}")
            logger.info(f"Device: {device_obj}, FP16: {use_fp16}")
            
            # 图输入维度固定为 num + cat
            num_output_dim = 64
            cat_output_dim = 32
            graph_input_dim = num_output_dim + cat_output_dim  # 96
            logger.info(f"Graph input dim: {graph_input_dim} (num={num_output_dim}, cat={cat_output_dim})")
            
            # 创建编码器
            num_encoder = NumericalEncoder(input_dim=8, hidden_dim=32, output_dim=num_output_dim).to(device_obj)
            cat_encoder = CategoricalEncoder(num_categories=[2,2,2,2,2], embedding_dim=16, output_dim=cat_output_dim).to(device_obj)
            graph_encoder = GraphEncoder(
                input_dim=graph_input_dim,
                hidden_dim=graph_hidden_dim,
                output_dim=graph_output_dim,
                num_relations=graph_num_relations,
                num_layers=graph_num_layers,
                dropout=0.0
            ).to(device_obj)
            
            num_encoder.eval()
            cat_encoder.eval()
            graph_encoder.eval()
            
            # 移动数据到设备
            num_features = num_features.to(device_obj)
            cat_features = cat_features.to(device_obj)
            edge_index = edge_index.to(device_obj)
            edge_type = edge_type.to(device_obj)
            
            # 计算嵌入
            with torch.no_grad():
                autocast_ctx = torch.amp.autocast('cuda') if (use_fp16 and device_obj.type == "cuda") else nullcontext()
                
                with autocast_ctx:
                    num_embed = num_encoder(num_features)
                    cat_embed = cat_encoder(cat_features)
                    graph_input = torch.cat([num_embed, cat_embed], dim=1)
                    embeddings = graph_encoder(graph_input.float(), edge_index, edge_type)
            
            embeddings = embeddings.float().cpu()
            elapsed = time.time() - start_time
            
            # 保存
            output_path = data_path / "graph_embeddings.pt"
            torch.save(embeddings, output_path)
            logger.info(f"Saved: {output_path} ({embeddings.shape})")
            logger.info(f"Computation time: {elapsed:.2f}s")
            
            # 保存元数据
            meta = {
                "graph_input_dim": graph_input_dim,
                "graph_hidden_dim": graph_hidden_dim,
                "graph_output_dim": graph_output_dim,
                "graph_num_relations": graph_num_relations,
                "graph_num_layers": graph_num_layers,
                "num_users": num_users
            }
            with open(data_path / "graph_embeddings_meta.json", 'w') as f:
                json.dump(meta, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="统一预处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python preprocess.py --dataset all --all          # 完整预处理
  python preprocess.py --dataset twibot20 --data    # 只预处理数据
  python preprocess.py --dataset all --text --graph # 预计算嵌入
  python preprocess.py -d all -a --batch-size 128   # 指定批次大小
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        choices=["twibot20", "misbot", "all"],
        help="要处理的数据集 (default: all)"
    )
    
    # 处理步骤
    parser.add_argument("--all", "-a", action="store_true", help="执行所有预处理步骤")
    parser.add_argument("--data", action="store_true", help="预处理原始数据")
    parser.add_argument("--text", action="store_true", help="预计算文本嵌入")
    parser.add_argument("--graph", action="store_true", help="预计算图嵌入")
    
    # 文本嵌入参数
    parser.add_argument("--text-model", type=str, default="xlm-roberta-base", help="文本模型 (default: xlm-roberta-base)")
    parser.add_argument("--max-length", type=int, default=128, help="最大token长度 (default: 128)")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小 (default: 64)")
    
    # 图嵌入参数
    parser.add_argument("--graph-hidden-dim", type=int, default=128, help="图隐藏层维度 (default: 128)")
    parser.add_argument("--graph-output-dim", type=int, default=128, help="图输出维度 (default: 128)")
    parser.add_argument("--graph-num-layers", type=int, default=2, help="RGCN层数 (default: 2)")
    
    # 通用参数
    parser.add_argument("--device", type=str, default="cuda", help="计算设备 (default: cuda)")
    parser.add_argument("--no-fp16", action="store_true", help="禁用FP16混合精度")
    parser.add_argument("--data-dir", type=str, default="processed_data", help="数据目录 (default: processed_data)")
    
    args = parser.parse_args()
    
    # 如果没有指定任何步骤，默认执行所有
    if not any([args.all, args.data, args.text, args.graph]):
        args.all = True
    
    # 确定数据集列表
    if args.dataset == "all":
        datasets = ["twibot20", "misbot"]
    else:
        datasets = [args.dataset]
    
    logger.info("="*60)
    logger.info("统一预处理脚本")
    logger.info("="*60)
    logger.info(f"数据集: {datasets}")
    logger.info(f"步骤: data={args.all or args.data}, text={args.all or args.text}, graph={args.all or args.graph}")
    logger.info(f"设备: {args.device}, FP16: {not args.no_fp16}")
    
    # 执行预处理步骤
    if args.all or args.data:
        preprocess_data(datasets, args.data_dir)
    
    if args.all or args.text:
        precompute_text_embeddings(
            datasets=datasets,
            data_dir=args.data_dir,
            model_name=args.text_model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            use_fp16=not args.no_fp16
        )
    
    if args.all or args.graph:
        precompute_graph_embeddings(
            datasets=datasets,
            data_dir=args.data_dir,
            graph_hidden_dim=args.graph_hidden_dim,
            graph_output_dim=args.graph_output_dim,
            graph_num_layers=args.graph_num_layers,
            device=args.device,
            use_fp16=not args.no_fp16
        )
    
    logger.info("\n" + "="*60)
    logger.info("预处理完成!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
