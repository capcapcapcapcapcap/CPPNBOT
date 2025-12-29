#!/usr/bin/env python3
"""
统一数据预处理脚本
融合Twibot作者经验与跨平台设计理念

使用方法:
    python run_preprocessing.py --config configs/preprocessing_config.yaml
    python run_preprocessing.py --dataset twibot20  # 只处理Twibot-20
    python run_preprocessing.py --dataset misbot    # 只处理Misbot
"""

import argparse
import yaml
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from data.unified_preprocessor import UnifiedPreprocessor

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='统一数据预处理')
    parser.add_argument('--config', type=str, default='configs/preprocessing_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str, choices=['twibot20', 'misbot', 'both'], 
                       default='both', help='要处理的数据集')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（覆盖配置文件）')
    parser.add_argument('--text_model', type=str, default=None,
                       help='文本编码模型（覆盖配置文件）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.text_model:
        config['text_encoding']['model_name'] = args.text_model
    
    # 创建预处理器
    preprocessor_config = {
        'output_dir': config['output_dir'],
        'text_model': config['text_encoding']['model_name'],
        'max_text_length': config['text_encoding']['max_length'],
        'text_dim': config['feature_dimensions']['text_dim'],
        'max_numerical_features': config['feature_dimensions']['numerical_dim'],
        'max_categorical_features': config['feature_dimensions']['categorical_dim']
    }
    
    preprocessor = UnifiedPreprocessor(preprocessor_config)
    
    print("=" * 60)
    print("🚀 统一数据预处理开始")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"处理数据集: {args.dataset}")
    print(f"输出目录: {config['output_dir']}")
    print(f"文本模型: {config['text_encoding']['model_name']}")
    print("=" * 60)
    
    results = {}
    
    # 处理Twibot-20
    if args.dataset in ['twibot20', 'both']:
        try:
            print("\n📊 处理 Twibot-20 数据集...")
            twibot_data = preprocessor.process_twibot20()
            twibot_files = preprocessor.save_processed_data(twibot_data, 'twibot20')
            
            results['twibot20'] = {
                'status': 'success',
                'num_users': len(twibot_data.user_ids),
                'files': twibot_files,
                'features': {
                    'text': twibot_data.user_texts is not None,
                    'numerical': twibot_data.numerical_features is not None,
                    'categorical': twibot_data.categorical_features is not None,
                    'graph': twibot_data.graph_edges is not None
                }
            }
            
            print(f"✅ Twibot-20 处理完成: {len(twibot_data.user_ids)} 用户")
            
        except Exception as e:
            print(f"❌ Twibot-20 处理失败: {e}")
            results['twibot20'] = {'status': 'failed', 'error': str(e)}
    
    # 处理Misbot
    if args.dataset in ['misbot', 'both']:
        try:
            print("\n📊 处理 Misbot 数据集...")
            misbot_data = preprocessor.process_misbot()
            misbot_files = preprocessor.save_processed_data(misbot_data, 'misbot')
            
            results['misbot'] = {
                'status': 'success',
                'num_users': len(misbot_data.user_ids),
                'files': misbot_files,
                'features': {
                    'text': misbot_data.user_texts is not None,
                    'numerical': misbot_data.numerical_features is not None,
                    'categorical': misbot_data.categorical_features is not None,
                    'graph': misbot_data.graph_edges is not None
                }
            }
            
            print(f"✅ Misbot 处理完成: {len(misbot_data.user_ids)} 用户")
            
        except Exception as e:
            print(f"❌ Misbot 处理失败: {e}")
            results['misbot'] = {'status': 'failed', 'error': str(e)}
    
    # 输出总结
    print("\n" + "=" * 60)
    print("📋 处理结果总结")
    print("=" * 60)
    
    for dataset, result in results.items():
        print(f"\n{dataset.upper()}:")
        if result['status'] == 'success':
            print(f"  ✅ 状态: 成功")
            print(f"  👥 用户数: {result['num_users']:,}")
            print(f"  📁 文件数: {len(result['files'])}")
            print(f"  🔤 文本特征: {'✓' if result['features']['text'] else '✗'}")
            print(f"  🔢 数值特征: {'✓' if result['features']['numerical'] else '✗'}")
            print(f"  🏷️  分类特征: {'✓' if result['features']['categorical'] else '✗'}")
            print(f"  🕸️  图结构: {'✓' if result['features']['graph'] else '✗'}")
        else:
            print(f"  ❌ 状态: 失败")
            print(f"  🐛 错误: {result['error']}")
    
    # 保存处理结果
    import json
    result_path = Path(config['output_dir']) / 'preprocessing_results.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 详细结果已保存到: {result_path}")
    print("\n🎉 数据预处理完成！")

if __name__ == "__main__":
    main()