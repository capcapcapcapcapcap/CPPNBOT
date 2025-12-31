# src/models/ - 模型定义模块

本目录包含原型网络模型的所有组件定义。

## 文件说明

### encoder.py

`MultiModalEncoder` 类 - 完整的多模态编码器。

组合数值编码器、分类编码器和融合模块，将原始特征映射到统一的嵌入空间。

**输入:** `{'num_features': [batch, 5], 'cat_features': [batch, 3]}`
**输出:** `Tensor[batch, 256]`

### fusion.py

`FusionModule` 类 - 多模态特征融合模块。

将数值嵌入 (64维) 和分类嵌入 (32维) 拼接后投影到 256 维。

**特点:**
- 拼接输入嵌入
- 线性投影
- ReLU 激活
- Dropout 正则化

### prototypical.py

`PrototypicalNetwork` 类 - 原型网络核心实现。

**主要方法:**
- `compute_prototypes(features, labels)`: 计算类原型 (类中心)
- `forward(support_set, query_set)`: 前向传播，返回 log 概率
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

**架构:** 5 → 32 → 64 (两层 MLP)
- Linear + ReLU
- Linear + LayerNorm

#### categorical.py

`CategoricalEncoder` 类 - 分类特征编码器。

**架构:**
- 每个分类特征独立的 Embedding 层
- 拼接所有嵌入
- 线性投影到 32 维

## 模型架构图

```
输入特征
    │
    ├── num_features [batch, 5]
    │       │
    │       ▼
    │   NumericalEncoder
    │       │
    │       ▼
    │   [batch, 64]
    │       │
    └── cat_features [batch, 3]
            │
            ▼
        CategoricalEncoder
            │
            ▼
        [batch, 32]
            │
            ▼
    ┌───────┴───────┐
    │  FusionModule │
    │   (64+32→256) │
    └───────┬───────┘
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

# 创建编码器
config = {
    'num_input_dim': 5,
    'num_hidden_dim': 32,
    'num_output_dim': 64,
    'cat_num_categories': [2, 2, 2],
    'cat_embedding_dim': 16,
    'cat_output_dim': 32,
    'fusion_output_dim': 256,
    'fusion_dropout': 0.1
}
encoder = MultiModalEncoder(config)

# 创建原型网络
model = PrototypicalNetwork(encoder, distance='euclidean')

# 前向传播
output = model(support_set, query_set)
predictions = output['log_probs'].argmax(dim=1)
```
