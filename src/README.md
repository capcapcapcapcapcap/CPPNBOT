# src/ - 源代码目录

本目录包含原型网络模型的核心实现代码。

## 目录结构

```
src/
├── __init__.py          # 包初始化，导出主要类
├── config/              # 配置管理模块
│   ├── __init__.py
│   └── config.py        # ModelConfig, TrainingConfig, Config 数据类
├── data/                # 数据加载模块
│   ├── __init__.py
│   ├── dataset.py       # BotDataset 数据集类
│   └── episode_sampler.py # EpisodeSampler N-way K-shot 采样器
├── models/              # 模型定义模块
│   ├── __init__.py
│   ├── encoders/        # 各类特征编码器
│   │   ├── __init__.py
│   │   ├── numerical.py   # NumericalEncoder (8→64)
│   │   ├── categorical.py # CategoricalEncoder (5→32)
│   │   ├── text.py        # TextEncoder (XLM-RoBERTa→256)
│   │   └── graph.py       # GraphEncoder (GAT→128)
│   ├── encoder.py       # MultiModalEncoder 多模态编码器
│   ├── fusion.py        # FusionModule, AttentionFusion 融合模块
│   └── prototypical.py  # PrototypicalNetwork 原型网络
└── training/            # 训练和评估模块
    ├── __init__.py
    ├── meta_trainer.py  # MetaTrainer 元训练器
    └── evaluator.py     # Evaluator 评估器
```

## 模块说明

### config/ - 配置管理

`config.py` 定义了三个数据类：
- `ModelConfig`: 模型架构配置（编码器维度、模态选择等）
- `TrainingConfig`: 训练配置（episode参数、学习率、早停等）
- `Config`: 完整配置（包含model和training）

支持从 YAML 文件加载配置，并进行参数验证。

### data/ - 数据处理

- `BotDataset`: 加载预处理后的机器人检测数据集
  - 加载数值特征、分类特征、标签、图结构
  - 提供 train/val/test 划分索引
  - 支持延迟加载用户文本

- `EpisodeSampler`: N-way K-shot Episode 采样器
  - 从数据集中采样 support 和 query 集
  - 确保类别平衡
  - 支持多模态数据索引追踪

### models/ - 模型定义

- `NumericalEncoder`: 数值特征编码器 (8→32→64)
  - 两层 MLP + ReLU + LayerNorm

- `CategoricalEncoder`: 分类特征编码器 (5→32)
  - Embedding 层 + 线性投影

- `TextEncoder`: 文本编码器 (XLM-RoBERTa→256)
  - 支持跨语言文本编码
  - 可选冻结骨干网络

- `GraphEncoder`: 图编码器 (RGCN→128)
  - 多层 RGCN 卷积
  - 原生支持多关系类型

- `MultiModalEncoder`: 多模态编码器
  - 组合各类编码器
  - 支持可配置模态组合

- `FusionModule`: 简单融合模块 (96→256)
  - 拼接 + 线性投影

- `AttentionFusion`: 注意力融合模块 (480→256)
  - 学习各模态权重
  - 加权融合

- `PrototypicalNetwork`: 原型网络核心
  - 计算类原型
  - 距离计算（欧氏/余弦）
  - 概率输出

### training/ - 训练评估

- `MetaTrainer`: 元训练器
  - Episode 训练循环
  - 验证和早停
  - 检查点保存
  - 多模态数据处理
  - 分离学习率支持

- `Evaluator`: 评估器
  - Few-shot 评估
  - 多 K-shot 值评估
  - 指标计算（Accuracy, Precision, Recall, F1）

## 使用示例

```python
from src.config import load_config
from src.data import BotDataset, EpisodeSampler
from src.models import MultiModalEncoder, PrototypicalNetwork
from src.training import MetaTrainer, Evaluator

# 加载配置
config = load_config("configs/default.yaml")

# 加载数据
dataset = BotDataset("twibot20", "processed_data")

# 创建模型
encoder = MultiModalEncoder(config.model.__dict__)
model = PrototypicalNetwork(encoder, distance=config.model.distance_metric)

# 训练
trainer = MetaTrainer(model, dataset, config.training.__dict__)
history = trainer.train()

# 评估
evaluator = Evaluator(model)
results = evaluator.evaluate_multiple_k_shots(
    dataset, 
    train_indices, 
    test_indices,
    k_shots=[1, 5, 10, 20]
)
```

## 多模态配置示例

```python
# 基线模型 (数值 + 分类)
config = {
    'enabled_modalities': ['num', 'cat'],
    'num_input_dim': 8,
    'cat_num_categories': [2, 2, 2, 2, 2],
    'fusion_output_dim': 256
}

# 加文本
config = {
    'enabled_modalities': ['num', 'cat', 'text'],
    'text_model_name': 'xlm-roberta-base',
    'text_freeze_backbone': True
}

# 完整模型
config = {
    'enabled_modalities': ['num', 'cat', 'text', 'graph'],
    'fusion_use_attention': True
}
```
