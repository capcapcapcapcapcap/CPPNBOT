# src/models/ - 模型定义模块

本目录包含原型网络模型的所有组件定义。

## 文件说明

### encoder.py

`MultiModalEncoder` 类 - 完整的多模态编码器。

支持可配置的模态组合，将原始特征映射到统一的嵌入空间。

**支持的模态:**
- `num`: 数值特征 (8维 → 64维)
- `cat`: 分类特征 (5维 → 32维)
- `text`: 文本特征 (XLM-RoBERTa → 256维)
- `graph`: 图特征 (GAT → 128维)

**输入:** `{'num_features': [batch, 8], 'cat_features': [batch, 5]}`
**输出:** `Tensor[batch, 256]`

### fusion.py

两种融合模块:

- `FusionModule`: 简单拼接融合
  - 拼接输入嵌入 (64+32=96)
  - 线性投影到 256 维
  - Dropout 正则化

- `AttentionFusion`: 注意力融合
  - 各模态投影到统一维度
  - 学习模态注意力权重
  - 加权融合 + LayerNorm

### prototypical.py

`PrototypicalNetwork` 类 - 原型网络核心实现。

**主要方法:**
- `compute_prototypes(features, labels)`: 计算类原型 (类中心)
- `forward(support_set, query_set, **kwargs)`: 前向传播，返回 log 概率
- `classify(support_set, query_features)`: 分类查询样本

**距离度量:**
- `euclidean`: 欧氏距离 (默认)
- `cosine`: 余弦距离

**输出:**
```python
{
    'log_probs': Tensor[n_query, n_classes],
    'prototypes': Tensor[n_classes, embed_dim],
    'query_embeddings': Tensor[n_query, embed_dim]
}
```

### encoders/ 子目录

#### numerical.py

`NumericalEncoder` 类 - 数值特征编码器。

**架构:** 8 → 32 → 64 (两层 MLP)
- Linear + ReLU
- Linear + LayerNorm

#### categorical.py

`CategoricalEncoder` 类 - 分类特征编码器。

**架构:**
- 每个分类特征独立的 Embedding 层 (2类 → 16维)
- 拼接所有嵌入 (5×16=80)
- 线性投影到 32 维

#### text.py

`TextEncoder` 类 - 文本编码器。

**架构:**
- XLM-RoBERTa 骨干网络
- 池化层 (CLS token)
- 线性投影到 256 维
- 支持骨干冻结

#### graph.py

`GraphEncoder` 类 - 图编码器。

**架构:**
- 多层 GAT 卷积
- 多头注意力机制
- 输出 128 维

## 模型架构图

```
输入特征
    │
    ├── num_features [batch, 8]
    │       │
    │       ▼
    │   NumericalEncoder (8→32→64)
    │       │
    │       ▼
    │   [batch, 64]
    │
    ├── cat_features [batch, 5]
    │       │
    │       ▼
    │   CategoricalEncoder (5→80→32)
    │       │
    │       ▼
    │   [batch, 32]
    │
    ├── texts (可选)
    │       │
    │       ▼
    │   TextEncoder (XLM-RoBERTa→256)
    │       │
    │       ▼
    │   [batch, 256]
    │
    └── graph (可选)
            │
            ▼
        GraphEncoder (GAT→128)
            │
            ▼
        [batch, 128]
            │
            ▼
    ┌───────────────────┐
    │  AttentionFusion  │
    │  (学习模态权重)    │
    │  → 256维          │
    └─────────┬─────────┘
              │
              ▼
          [batch, 256]
              │
              ▼
      PrototypicalNetwork
              │
              ▼
      log_probs [n_query, n_classes]
```

## 使用示例

```python
from src.models import MultiModalEncoder, PrototypicalNetwork

# 创建编码器 (基线配置)
config = {
    'num_input_dim': 8,
    'num_hidden_dim': 32,
    'num_output_dim': 64,
    'cat_num_categories': [2, 2, 2, 2, 2],
    'cat_embedding_dim': 16,
    'cat_output_dim': 32,
    'fusion_output_dim': 256,
    'fusion_dropout': 0.1,
    'fusion_use_attention': True,
    'enabled_modalities': ['num', 'cat']
}
encoder = MultiModalEncoder(config)

# 创建原型网络
model = PrototypicalNetwork(encoder, distance='euclidean')

# 前向传播
output = model(support_set, query_set)
predictions = output['log_probs'].argmax(dim=1)

# 获取注意力权重 (如果使用注意力融合)
weights = encoder.get_attention_weights()
print(weights)  # {'num': 0.6, 'cat': 0.4}
```
