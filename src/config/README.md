# src/config/ - 配置管理模块

本目录包含实验配置管理相关代码。

## 文件说明

### config.py

定义配置数据类和配置加载函数。

**数据类:**

- `ModelConfig`: 模型架构参数
  - 数值编码器: `num_input_dim` (8), `num_hidden_dim` (32), `num_output_dim` (64)
  - 分类编码器: `cat_num_categories` ([2,2,2,2,2]), `cat_embedding_dim` (16), `cat_output_dim` (32)
  - 文本编码器: `text_model_name`, `text_output_dim` (256), `text_max_length` (512), `text_freeze_backbone`
  - 图编码器 (RGCN): `graph_input_dim`, `graph_hidden_dim`, `graph_output_dim`, `graph_num_relations`, `graph_num_layers`, `graph_num_bases`
  - 融合模块: `fusion_output_dim` (256), `fusion_dropout` (0.1), `fusion_use_attention`
  - 模态配置: `enabled_modalities` (['num', 'cat'])
  - 距离度量: `distance_metric` ('euclidean' 或 'cosine')

- `TrainingConfig`: 训练参数
  - Episode 配置: `n_way` (2), `k_shot` (10), `n_query` (15)
  - Episode 数量: `n_episodes_train` (100), `n_episodes_val` (50)
  - 训练参数: `n_epochs` (200), `learning_rate` (0.001), `weight_decay` (0.0001)
  - 文本学习率: `text_learning_rate` (0.00001)
  - 早停: `patience` (10)

- `Config`: 完整配置
  - `model`: ModelConfig 实例
  - `training`: TrainingConfig 实例
  - `data_dir`: 数据目录路径
  - `output_dir`: 输出目录路径
  - `seed`: 随机种子

**函数:**
- `load_config(path: str) -> Config`: 从 YAML 文件加载配置

## 使用示例

```python
from src.config import load_config, Config, ModelConfig, TrainingConfig

# 从文件加载
config = load_config("configs/default.yaml")

# 访问参数
print(config.model.fusion_output_dim)  # 256
print(config.model.enabled_modalities)  # ['num', 'cat']
print(config.training.k_shot)          # 10

# 命令行覆盖
config.training.n_epochs = 50
```

## 配置验证

`ModelConfig` 和 `TrainingConfig` 在 `__post_init__` 中进行参数验证:
- 维度参数必须为正整数
- Dropout 必须在 [0, 1) 范围内
- 距离度量必须是 'euclidean' 或 'cosine'
- 模态列表不能为空
