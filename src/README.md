# src/ - 源代码目录

本目录包含原型网络模型的核心实现代码。

## 目录结构

```
src/
├── __init__.py          # 包初始化文件
├── config/              # 配置管理模块
├── data/                # 数据加载模块
├── models/              # 模型定义模块
└── training/            # 训练和评估模块
```

## 模块说明

### config/ - 配置管理
- `config.py`: 定义 `ModelConfig`, `TrainingConfig`, `Config` 数据类，实现 YAML 配置加载和参数验证

### data/ - 数据处理
- `dataset.py`: `BotDataset` 类，加载预处理后的机器人检测数据集
- `episode_sampler.py`: `EpisodeSampler` 类，实现 N-way K-shot Episode 采样

### models/ - 模型定义
- `encoder.py`: `MultiModalEncoder` 多模态编码器，组合数值和分类特征编码器
- `fusion.py`: `FusionModule` 特征融合模块
- `prototypical.py`: `PrototypicalNetwork` 原型网络核心实现
- `encoders/`: 子模块，包含各类特征编码器
  - `numerical.py`: 数值特征编码器 (5维 → 64维)
  - `categorical.py`: 分类特征编码器 (3维 → 32维)

### training/ - 训练评估
- `meta_trainer.py`: `MetaTrainer` 元训练器，实现 episode 训练循环
- `evaluator.py`: `Evaluator` 评估器，实现少样本评估和指标计算

## 使用示例

```python
from src.config import load_config
from src.data import BotDataset, EpisodeSampler
from src.models import MultiModalEncoder, PrototypicalNetwork
from src.training import MetaTrainer, Evaluator

# 加载配置
config = load_config("configs/default.yaml")

# 加载数据
dataset = BotDataset("twibot20")

# 创建模型
encoder = MultiModalEncoder(config.model)
model = PrototypicalNetwork(encoder)

# 训练
trainer = MetaTrainer(model, dataset, config.training)
trainer.train()
```
