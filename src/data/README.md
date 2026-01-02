# src/data/ - 数据加载模块

本目录包含数据集加载和 Episode 采样相关代码。

## 文件说明

### dataset.py

`BotDataset` 类 - 加载预处理后的机器人检测数据集。

**主要方法:**
- `__init__(dataset_name, data_dir)`: 初始化并加载所有预处理张量
- `__len__()`: 返回数据集用户数量
- `__getitem__(idx)`: 返回单个用户的特征字典
- `get_split_indices(split)`: 获取 train/val/test 划分索引
- `get_user_texts()`: 加载用户文本数据 (延迟加载)

**加载的数据:**
- `num_features`: 数值特征张量 [N, 8]
- `cat_features`: 分类特征张量 [N, 5]
- `labels`: 标签张量 [N] (0=human, 1=bot, -1=unknown)
- `edge_index`: 图边索引 [2, E]
- `edge_type`: 边类型 [E]
- `train_idx/val_idx/test_idx`: 数据划分索引

### episode_sampler.py

`EpisodeSampler` 类 - N-way K-shot Episode 采样器。

**主要方法:**
- `__init__(n_way, k_shot, n_query)`: 初始化采样器参数
- `sample(dataset, indices)`: 从指定索引采样一个 episode

**返回格式:**
```python
support_set = {
    'num_features': Tensor[n_way*k_shot, 8],
    'cat_features': Tensor[n_way*k_shot, 5],
    'labels': Tensor[n_way*k_shot]
}
query_set = {
    'num_features': Tensor[n_way*n_query, 8],
    'cat_features': Tensor[n_way*n_query, 5],
    'labels': Tensor[n_way*n_query]
}
```

**索引追踪:**
采样后可通过 `sampler._last_support_indices` 和 `sampler._last_query_indices` 获取原始索引，用于多模态数据加载。

**异常类:**
- `InsufficientSamplesError`: 当某类别样本不足时抛出

## 使用示例

```python
from src.data import BotDataset, EpisodeSampler

# 加载数据集
dataset = BotDataset("twibot20", "processed_data")
print(f"数据集大小: {len(dataset)}")
print(f"数值特征维度: {dataset.num_features.shape}")  # [N, 8]
print(f"分类特征维度: {dataset.cat_features.shape}")  # [N, 5]

# 获取训练集索引
train_idx = dataset.get_split_indices("train")

# 创建采样器
sampler = EpisodeSampler(n_way=2, k_shot=10, n_query=15)

# 采样一个 episode
support, query = sampler.sample(dataset, train_idx)

# 获取用户文本 (用于文本模态)
texts = dataset.get_user_texts()
print(texts[0])  # {'description': '...', 'tweets': [...]}
```
