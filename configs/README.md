# configs/ - 配置文件目录

本目录包含实验配置文件。

## 文件说明

### default.yaml

默认配置文件，包含所有参数的默认值。

**配置结构:**
```yaml
# 模型配置
model:
  num_input_dim: 5          # 数值特征输入维度
  num_hidden_dim: 32        # 数值编码器隐藏层维度
  num_output_dim: 64        # 数值编码器输出维度
  cat_num_categories:       # 各分类特征的类别数
    - 2
    - 2
    - 2
  cat_embedding_dim: 16     # 分类嵌入维度
  cat_output_dim: 32        # 分类编码器输出维度
  fusion_output_dim: 256    # 融合后输出维度
  fusion_dropout: 0.1       # Dropout 比率
  distance_metric: euclidean  # 距离度量 (euclidean/cosine)

# 训练配置
training:
  n_way: 2                  # Episode 类别数
  k_shot: 5                 # 每类 support 样本数
  n_query: 15               # 每类 query 样本数
  n_episodes_train: 100     # 每 epoch 训练 episode 数
  n_episodes_val: 50        # 验证 episode 数
  n_epochs: 100             # 最大训练轮数
  learning_rate: 0.001      # 学习率
  weight_decay: 0.0001      # 权重衰减
  patience: 10              # 早停耐心值

# 路径配置
data_dir: processed_data    # 数据目录
output_dir: results         # 输出目录
seed: 42                    # 随机种子
```

## 创建自定义配置

1. 复制默认配置:
```bash
cp configs/default.yaml configs/custom.yaml
```

2. 修改需要的参数:
```yaml
# configs/custom.yaml
model:
  distance_metric: cosine   # 使用余弦距离

training:
  k_shot: 10                # 10-shot 学习
  n_epochs: 200             # 更多训练轮数
  learning_rate: 0.0005     # 更小的学习率
```

3. 使用自定义配置:
```bash
python experiments/train_source.py --config configs/custom.yaml
```

## 配置覆盖优先级

命令行参数 > 配置文件 > 默认值

```bash
# 配置文件中 epochs=100，但命令行覆盖为 50
python experiments/train_source.py --config configs/default.yaml --epochs 50
```

## 推荐配置

### 快速验证
```yaml
training:
  n_episodes_train: 20
  n_episodes_val: 10
  n_epochs: 5
```

### 完整训练
```yaml
training:
  n_episodes_train: 200
  n_episodes_val: 100
  n_epochs: 200
  patience: 20
```

### 少样本实验
```yaml
training:
  k_shot: 1   # 1-shot
  # 或
  k_shot: 5   # 5-shot
  # 或
  k_shot: 10  # 10-shot
```
