# 基于原型网络的跨平台社交机器人检测

## 1. 研究问题

**核心问题**：如何让机器人检测模型快速适应新的社交平台？

**现实挑战**：
- 不同平台数据格式、语言、用户行为差异大
- 新平台标注数据稀缺且获取成本高
- 传统监督学习需要大量标注数据，迁移效果差

**研究目标**：
- 在源域（Twibot-20）学习"什么是机器人"的通用模式
- 在目标域（Misbot）仅用5-10个标注样本快速适应
- 验证跨语言、跨平台的少样本检测能力

---

## 2. 方法设计

### 2.1 为什么选择原型网络

| 方法 | 优势 | 劣势 | 适用性 |
|------|------|------|--------|
| 微调 | 简单 | 需要大量目标域数据 | ❌ |
| MAML | 快速适应 | 二阶优化，训练不稳定 | △ |
| **原型网络** | 简单有效，无需微调 | 依赖好的特征表示 | ✅ |

原型网络的核心思想：**同类样本在特征空间中聚集，不同类样本分离**。
- 计算每个类别的"原型"（类中心）
- 新样本根据到各原型的距离分类
- 天然适合少样本场景

### 2.2 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      元训练阶段 (Twibot-20)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Episode采样 → 特征编码器 → 原型计算 → 距离分类      │
│   (N-way K-shot)   (多模态)      (类中心)    (查询集)       │
│                                                             │
│   重复数千个episode，学习：                                  │
│   1. 如何从少量样本提取有效特征                              │
│   2. 如何构建有区分度的类原型                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
                        保存编码器权重
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      少样本适应阶段 (Misbot)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Support Set → 预训练编码器 → 计算新原型 → 分类      │
│   (5个human,       (冻结)         (目标域)      (测试集)    │
│    5个bot)                                                  │
│                                                             │
│   无需训练，直接用support set构建原型进行预测                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 特征设计

**关键决策**：使用什么特征？

经过分析两个数据集的共有信息：

| 特征类型 | Twibot-20 | Misbot | 统一方案 |
|----------|-----------|--------|----------|
| 数值特征 | followers, following, tweets, listed, age | followers, following, tweets | **5维统一**（缺失填0） |
| 分类特征 | verified, protected, default_avatar | 20维categorical | **3维统一**（取前3维） |
| 文本特征 | description + tweets (英文) | description + tweets (中文) | **跨语言编码器** |
| 图结构 | follow, friend关系 | follow, mention关系 | **统一边类型** |

**特征编码器设计**：

```python
用户表示 = Fusion(
    数值编码(5维 → 64维),      # MLP
    分类编码(3维 → 32维),      # Embedding
    文本编码(描述+推文 → 256维), # XLM-RoBERTa (跨语言)
    图编码(邻居聚合 → 128维)    # GAT (可选)
)
# 最终: 480维 → 256维 (投影层)
```

---

## 3. 数据处理

### 3.1 预处理目标

将两个数据集转换为**统一格式**，供模型直接使用：

```
processed_data/
├── twibot20/
│   ├── users.pt          # 用户基础信息 (id, 原始特征)
│   ├── num_features.pt   # 数值特征 [N, 5] (已标准化)
│   ├── cat_features.pt   # 分类特征 [N, 3]
│   ├── labels.pt         # 标签 [N] (0=human, 1=bot, -1=未标注)
│   ├── splits.pt         # 划分索引 {train, val, test}
│   ├── edge_index.pt     # 图边 [2, E]
│   ├── edge_type.pt      # 边类型 [E]
│   └── user_tweets.npy   # 用户-推文映射 {user_idx: [tweet_ids]}
│
└── misbot/
    └── (同上结构)
```

### 3.2 预处理流程

```
原始数据 → 流式加载 → 特征提取 → 标准化 → 保存
           (解决内存)    (统一格式)   (Z-score)
```

### 3.3 文本处理策略

**预处理阶段**：只保存原始文本，不做编码
**模型阶段**：在线编码，支持端到端训练

原因：
1. 文本编码器是模型的一部分，应该可以微调
2. 避免预处理和模型的耦合
3. 不同实验可能用不同的文本模型

---

## 4. 模型实现

### 4.1 核心组件

```
src/
├── data/
│   ├── dataset.py        # 数据集类
│   └── episode_sampler.py # Episode采样器
│
├── models/
│   ├── encoders/
│   │   ├── numerical.py  # 数值编码器
│   │   ├── categorical.py # 分类编码器
│   │   ├── text.py       # 文本编码器 (XLM-RoBERTa)
│   │   └── graph.py      # 图编码器 (GAT)
│   │
│   ├── fusion.py         # 多模态融合
│   └── prototypical.py   # 原型网络
│
└── training/
    ├── meta_trainer.py   # 元训练循环
    └── evaluator.py      # 评估器
```

### 4.2 原型网络核心逻辑

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder, distance='euclidean'):
        self.encoder = encoder  # 多模态编码器
        self.distance = distance
    
    def forward(self, support_x, support_y, query_x):
        # 1. 编码所有样本
        support_features = self.encoder(support_x)  # [N_support, D]
        query_features = self.encoder(query_x)      # [N_query, D]
        
        # 2. 计算类原型 (每个类的特征均值)
        prototypes = {}
        for label in [0, 1]:  # human, bot
            mask = (support_y == label)
            prototypes[label] = support_features[mask].mean(dim=0)
        
        # 3. 计算查询样本到各原型的距离
        proto_stack = torch.stack([prototypes[0], prototypes[1]])  # [2, D]
        distances = torch.cdist(query_features, proto_stack)       # [N_query, 2]
        
        # 4. 距离转概率 (距离越小，概率越大)
        logits = -distances
        return F.log_softmax(logits, dim=-1)
```

### 4.3 Episode采样

```python
class EpisodeSampler:
    """从数据集中采样N-way K-shot episode"""
    
    def __init__(self, n_way=2, k_shot=5, n_query=15):
        self.n_way = n_way      # 类别数 (human vs bot)
        self.k_shot = k_shot    # 每类支持样本数
        self.n_query = n_query  # 每类查询样本数
    
    def sample(self, dataset, indices):
        """
        从indices中采样一个episode
        返回: support_set, query_set
        """
        # 按类别分组
        human_idx = [i for i in indices if dataset.labels[i] == 0]
        bot_idx = [i for i in indices if dataset.labels[i] == 1]
        
        # 采样support set
        support_human = random.sample(human_idx, self.k_shot)
        support_bot = random.sample(bot_idx, self.k_shot)
        
        # 采样query set (不与support重叠)
        remaining_human = [i for i in human_idx if i not in support_human]
        remaining_bot = [i for i in bot_idx if i not in support_bot]
        query_human = random.sample(remaining_human, self.n_query)
        query_bot = random.sample(remaining_bot, self.n_query)
        
        return (support_human + support_bot, 
                query_human + query_bot)
```

---

## 5. 实验设计

### 5.1 实验一：源域内验证

**目的**：验证原型网络在单一数据集上的有效性

**设置**：
- 数据集：Twibot-20
- 训练：train split上采样episode
- 验证：val split上评估
- 测试：test split上报告最终结果

**对比基线**：
- MLP分类器（全量监督）
- 随机森林（传统ML）
- 5-shot微调

### 5.2 实验二：跨域迁移（核心实验）

**目的**：验证跨平台少样本适应能力

**设置**：
- 元训练：Twibot-20全部数据
- 适应：Misbot的K个标注样本（K=1,5,10,20）
- 测试：Misbot测试集

**评估指标**：
- Accuracy, Precision, Recall, F1
- 不同K值下的性能曲线

**对比基线**：
- 直接迁移（不适应）
- 微调（用K个样本微调）
- 从头训练（只用K个样本）

### 5.3 实验三：消融实验

**目的**：分析各组件贡献

| 实验 | 设置 | 验证问题 |
|------|------|----------|
| A | 去掉文本特征 | 文本的重要性 |
| B | 去掉图结构 | 图的重要性 |
| C | 只用数值特征 | 最小特征集效果 |
| D | 不同距离度量 | 欧氏 vs 余弦 |
| E | 不同K-shot | 样本数敏感性 |

---

## 6. 实施计划

### 第一阶段：数据准备（1周）
- [x] 统一预处理器实现
- [x] 流式加载优化
- [x] 运行预处理，验证输出

### 第二阶段：基础模型（2周）
- [x] 数值/分类编码器
- [x] 原型网络核心
- [x] Episode采样器
- [ ] 源域内验证实验

### 第三阶段：完整模型（2周）
- [ ] 文本编码器（XLM-RoBERTa）
- [ ] 图编码器（GAT）
- [ ] 多模态融合
- [ ] 跨域迁移实验

### 第四阶段：实验与论文（2周）
- [ ] 消融实验
- [ ] 结果分析
- [ ] 论文撰写

---

## 7. 预期贡献

1. **方法贡献**：首次将原型网络应用于跨平台社交机器人检测
2. **实验贡献**：建立Twibot-20→Misbot跨域评估基准
3. **实用价值**：提供少样本快速部署方案

---

## 8. 项目结构

```
CPPNBOT/
├── dataset/                    # 原始数据
│   ├── Twibot-20/
│   └── Misbot/
│
├── processed_data/             # 预处理后数据
│   ├── twibot20/
│   └── misbot/
│
├── src/                        # 源代码
│   ├── data/
│   ├── models/
│   └── training/
│
├── experiments/                # 实验脚本
│   ├── train_source.py        # 源域训练
│   ├── adapt_target.py        # 目标域适应
│   └── ablation.py            # 消融实验
│
├── configs/                    # 配置文件
├── results/                    # 实验结果
│
├── preprocess_unified.py       # 预处理脚本
├── DESIGN_DOCUMENT.md          # 本文档（实验设计）
└── requirements.txt
```
