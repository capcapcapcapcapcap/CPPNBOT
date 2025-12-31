# src/config/ - 配置管理模块

本目录包含实验配置管理相关代码。

## 文件说明

### config.py

定义配置数据类和配置加载函数。

**数据类:**
- `ModelConfig`: 模型架构参数
  - `num_input_dim`: 数值特征输入维度 (默认: 5)
  - `num_hidden_dim`: 数值编码器隐藏层维度 (默认: 32)
  - `num_output_dim`: 数值编码器输出维度 (默认: 64)
  - `cat_num_categories`: 各分类特征的类别数
  - `cat_embedding_dim`: 分类嵌入维度 (默认: 16)
  - `cat_output_dim`: 分类编码器输出维度 (默认: 32)
  - `fusion_output_dim`: 融合后输出维度 (默认: 256)
  - `fusion_dropout`: Dropout 比率 (默认: 0.1)
  - `distance_metric`: 距离度量 ('euclidean' 或 'cosine')

- `TrainingConfig`: 训练参数
  - `n_way`: Episode 类别数 (默认: 2)
  - `k_shot`: 每类 support 样本数 (默认: 5)
  - `n_query`: 每类 query 样本数 (默认: 15)
  - `n_episodes_train`: 每 epoch 训练 episode 数 (默认: 100)
  - `n_episodes_val`: 验证 episode 数 (默认: 50)
  - `n_epochs`: 最大训练轮数 (默认: 100)
  - `learning_rate`: 学习率 (默认: 1e-3)
  - `weight_decay`: 权重衰减 (默认: 1e-4)
  - `patience`: 早停耐心值 (默认: 10)

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
print(config.training.k_shot)          # 5

# 命令行覆盖
config.training.n_epochs = 50
```
