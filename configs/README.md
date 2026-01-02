# configs/ - 配置文件目录

本目录包含实验配置文件。

## 文件说明

| 文件 | 模态 | 说明 |
|------|------|------|
| default.yaml | num + cat | 默认配置，基线模型 |
| ablation_num_cat.yaml | num + cat | 消融实验：基线 |
| ablation_num_cat_text.yaml | num + cat + text | 消融实验：+文本 |
| ablation_num_cat_graph.yaml | num + cat + graph | 消融实验：+图 (RGCN) |
| ablation_all.yaml | num + cat + text + graph | 消融实验：完整模型 |

## 配置结构

```yaml
# 模型配置
model:
  # 数值编码器
  num_input_dim: 8          # 数值特征输入维度
  num_hidden_dim: 32        # 数值编码器隐藏层维度
  num_output_dim: 64        # 数值编码器输出维度
  
  # 分类编码器
  cat_num_categories: [2, 2, 2, 2, 2]  # 各分类特征的类别数
  cat_embedding_dim: 16     # 分类嵌入维度
  cat_output_dim: 32        # 分类编码器输出维度
  
  # 文本编码器 (可选)
  text_model_name: xlm-roberta-base
  text_output_dim: 256
  text_max_length: 128      # 减小以加速处理
  text_freeze_backbone: true
  use_precomputed_text_embeddings: true  # 使用预计算嵌入加速训练
  
  # 图编码器 (RGCN，可选)
  graph_input_dim: 256
  graph_hidden_dim: 128
  graph_output_dim: 128
  graph_num_relations: 2    # 边类型数量 (follow, friend)
  graph_num_layers: 2       # RGCN层数
  graph_dropout: 0.1
  
  # 融合模块
  fusion_output_dim: 256    # 融合后输出维度
  fusion_dropout: 0.1       # Dropout 比率
  fusion_use_attention: true # 使用注意力融合
  
  # 启用的模态
  enabled_modalities: ['num', 'cat']  # 可选: 'text', 'graph'
  
  # 距离度量
  distance_metric: euclidean  # 或 'cosine'

# 训练配置
training:
  n_way: 2                  # Episode 类别数
  k_shot: 10                # 每类 support 样本数
  n_query: 15               # 每类 query 样本数
  n_episodes_train: 100     # 每 epoch 训练 episode 数
  n_episodes_val: 50        # 验证 episode 数
  n_epochs: 200             # 最大训练轮数
  learning_rate: 0.001      # 主学习率
  text_learning_rate: 0.00001  # 文本编码器学习率
  weight_decay: 0.0001      # 权重衰减
  patience: 10              # 早停耐心值

# 路径配置
data_dir: processed_data    # 数据目录
output_dir: results         # 输出目录

# 随机种子 (null = 随机)
seed: null
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
  enabled_modalities: ['num', 'cat', 'text']  # 加文本

training:
  k_shot: 5                 # 5-shot 学习
  n_epochs: 100             # 更少训练轮数
  learning_rate: 0.0005     # 更小的学习率
```

3. 使用自定义配置:
```bash
python experiments/train_source.py --config configs/custom.yaml
```

## 配置覆盖优先级

命令行参数 > 配置文件 > 默认值

```bash
# 配置文件中 epochs=200，但命令行覆盖为 50
python experiments/train_source.py --config configs/default.yaml --epochs 50
```

## 推荐配置

### 快速验证
```yaml
training:
  n_episodes_train: 20
  n_episodes_val: 10
  n_epochs: 5
  patience: 3
```

### 完整训练
```yaml
training:
  n_episodes_train: 100
  n_episodes_val: 50
  n_epochs: 200
  patience: 10
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

### 多模态实验
```yaml
model:
  # 基线
  enabled_modalities: ['num', 'cat']
  
  # 加文本 (使用预计算嵌入)
  enabled_modalities: ['num', 'cat', 'text']
  use_precomputed_text_embeddings: true  # 推荐，大幅加速训练
  
  # 加文本 (在线编码，较慢)
  enabled_modalities: ['num', 'cat', 'text']
  use_precomputed_text_embeddings: false
  text_freeze_backbone: true
  
  # 完整模型
  enabled_modalities: ['num', 'cat', 'text', 'graph']
  fusion_use_attention: true
```

## 预计算嵌入

为了加速包含文本或图模态的训练，建议预先计算嵌入：

### 文本嵌入
```bash
# 预计算所有数据集的文本嵌入
python precompute_text_embeddings.py --dataset all

# 或单独处理
python precompute_text_embeddings.py --dataset twibot20
python precompute_text_embeddings.py --dataset misbot
```

### 图嵌入
```bash
# 预计算所有数据集的图嵌入
python precompute_graph_embeddings.py --dataset all

# 或单独处理
python precompute_graph_embeddings.py --dataset twibot20
python precompute_graph_embeddings.py --dataset misbot
```

图编码器输入固定为 num + cat 特征 (96维)，与文本模态独立。各模态在融合层进行交互。

预计算后，训练时会自动使用预计算的嵌入文件。

**性能对比**:
- 在线编码: ~30秒/epoch (Transformer/RGCN 推理开销大)
- 预计算嵌入: ~0.1秒/epoch (直接查表)

## 模态组合说明

| 模态 | 编码器 | 输出维度 | 说明 |
|------|--------|----------|------|
| num | NumericalEncoder | 64 | 8维数值特征 |
| cat | CategoricalEncoder | 32 | 5维分类特征 |
| text | TextEncoder (XLM-RoBERTa) 或预计算嵌入 | 256 | 用户描述+推文 |
| graph | GraphEncoder (RGCN) | 128 | 社交网络结构，支持多关系类型 |

融合后统一输出 256 维嵌入向量。
