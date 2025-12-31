# src/models/encoders/ - 特征编码器模块

本目录包含各类特征的编码器实现。

## 文件说明

### numerical.py

`NumericalEncoder` 类 - 数值特征编码器。

将 5 维数值特征（如关注数、粉丝数等统计信息）编码为 64 维稠密向量。

**架构:**
```
Input [batch, 5]
    │
    ▼
Linear(5, 32)
    │
    ▼
ReLU
    │
    ▼
Linear(32, 64)
    │
    ▼
LayerNorm(64)
    │
    ▼
Output [batch, 64]
```

**参数:**
- `input_dim`: 输入维度 (默认: 5)
- `hidden_dim`: 隐藏层维度 (默认: 32)
- `output_dim`: 输出维度 (默认: 64)

### categorical.py

`CategoricalEncoder` 类 - 分类特征编码器。

将 3 维分类特征（如账户验证状态、默认头像等）编码为 32 维稠密向量。

**架构:**
```
Input [batch, 3] (整数索引)
    │
    ├── Feature 0 → Embedding(n_cat_0, 16)
    ├── Feature 1 → Embedding(n_cat_1, 16)
    └── Feature 2 → Embedding(n_cat_2, 16)
            │
            ▼
        Concatenate
            │
            ▼
        [batch, 48]
            │
            ▼
        Linear(48, 32)
            │
            ▼
        Output [batch, 32]
```

**参数:**
- `num_categories`: 每个特征的类别数列表
- `embedding_dim`: 每个特征的嵌入维度 (默认: 16)
- `output_dim`: 输出维度 (默认: 32)

## 使用示例

```python
import torch
from src.models.encoders import NumericalEncoder, CategoricalEncoder

# 数值编码器
num_encoder = NumericalEncoder(input_dim=5, output_dim=64)
num_features = torch.randn(32, 5)  # batch=32
num_embed = num_encoder(num_features)  # [32, 64]

# 分类编码器
cat_encoder = CategoricalEncoder(
    num_categories=[2, 2, 2],
    embedding_dim=16,
    output_dim=32
)
cat_features = torch.randint(0, 2, (32, 3))  # batch=32
cat_embed = cat_encoder(cat_features)  # [32, 32]
```

## 设计说明

1. **LayerNorm**: 数值编码器使用 LayerNorm 而非 BatchNorm，因为元学习中 batch 大小可能很小

2. **独立嵌入表**: 分类编码器为每个特征使用独立的嵌入表，允许不同特征有不同的类别数

3. **维度选择**: 
   - 数值特征信息量较大，编码到 64 维
   - 分类特征信息量较小，编码到 32 维
   - 融合后总维度为 256，提供足够的表达能力
