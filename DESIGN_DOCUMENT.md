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
│   Episode采样 → 多模态编码器 → 原型计算 → 距离分类           │
│   (2-way 10-shot)  (可配置模态)   (类中心)    (查询集)       │
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
│   Support Set → 预训练编码器 → 计算新原型 → 分类测试集       │
│   (K个human,       (冻结)         (目标域)                  │
│    K个bot)                                                  │
│                                                             │
│   无需训练，直接用support set构建原型进行预测                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 特征设计

**多模态特征体系**：

| 特征类型 | 维度 | 编码器 | 说明 |
|----------|------|--------|------|
| 数值特征 | 8维 → 64维 | NumericalEncoder (MLP) | followers, following, tweets, listed, age, ratio, username_len, desc_len |
| 分类特征 | 5维 → 32维 | CategoricalEncoder (Embedding) | verified, protected, default_avatar, has_url, has_location |
| 文本特征 | 文本 → 256维 | TextEncoder (XLM-RoBERTa) | description + tweets，支持跨语言 |
| 图特征 | 节点 → 128维 | GraphEncoder (GAT) | 社交网络结构信息 |

**特征融合**：
- 基础模式：简单拼接 + 投影 (96维 → 256维)
- 注意力模式：学习各模态权重，加权融合 (480维 → 256维)

**模态组合配置**：
```yaml
# 基线 (默认)
enabled_modalities: ['num', 'cat']

# 加文本
enabled_modalities: ['num', 'cat', 'text']

# 完整模型
enabled_modalities: ['num', 'cat', 'text', 'graph']
```

---

## 3. 数据处理

### 3.1 预处理目标

将两个数据集转换为**统一格式**，供模型直接使用：

```
processed_data/
├── twibot20/
│   ├── num_features.pt   # 数值特征 [N, 8] (已标准化)
│   ├── cat_features.pt   # 分类特征 [N, 5]
│   ├── labels.pt         # 标签 [N] (0=human, 1=bot, -1=未标注)
│   ├── train_idx.pt      # 训练集索引
│   ├── val_idx.pt        # 验证集索引
│   ├── test_idx.pt       # 测试集索引
│   ├── edge_index.pt     # 图边 [2, E]
│   ├── edge_type.pt      # 边类型 [E]
│   ├── user_texts.json   # 用户文本 {idx: {description, tweets}}
│   └── metadata.json     # 元数据
│
└── misbot/
    └── (同上结构)
```

### 3.2 数值特征 (8维)

| 索引 | 特征名 | 说明 |
|------|--------|------|
| 0 | followers_count | 粉丝数 |
| 1 | following_count | 关注数 |
| 2 | tweet_count | 推文数 |
| 3 | listed_count | 被列表收录次数 |
| 4 | account_age_days | 账户年龄天数 |
| 5 | followers_following_ratio | 粉丝/关注比 |
| 6 | username_length | 用户名长度 |
| 7 | description_length | 简介长度 |

### 3.3 分类特征 (5维)

| 索引 | 特征名 | 类别 |
|------|--------|------|
| 0 | verified | 0=未验证, 1=已验证 |
| 1 | protected | 0=公开, 1=受保护 |
| 2 | default_avatar | 0=自定义头像, 1=默认头像 |
| 3 | has_url | 0=无URL, 1=有URL |
| 4 | has_location | 0=无位置, 1=有位置 |

### 3.4 预处理流程

```
原始数据 → 流式加载 → 特征提取 → 标准化 → 保存
           (解决内存)    (统一格式)   (Z-score)
```

---

## 4. 模型实现

### 4.1 核心组件

```
src/
├── config/
│   └── config.py         # ModelConfig, TrainingConfig, Config 数据类
│
├── data/
│   ├── dataset.py        # BotDataset 数据集类
│   └── episode_sampler.py # EpisodeSampler N-way K-shot 采样器
│
├── models/
│   ├── encoders/
│   │   ├── numerical.py  # NumericalEncoder (8→64)
│   │   ├── categorical.py # CategoricalEncoder (5→32)
│   │   ├── text.py       # TextEncoder (XLM-RoBERTa→256)
│   │   └── graph.py      # GraphEncoder (GAT→128)
│   │
│   ├── encoder.py        # MultiModalEncoder 多模态编码器
│   ├── fusion.py         # FusionModule, AttentionFusion 融合模块
│   └── prototypical.py   # PrototypicalNetwork 原型网络
│
└── training/
    ├── meta_trainer.py   # MetaTrainer 元训练器
    └── evaluator.py      # Evaluator 评估器
```

### 4.2 原型网络核心逻辑

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder, distance='euclidean'):
        self.encoder = encoder  # 多模态编码器
        self.distance = distance
    
    def forward(self, support_set, query_set, **kwargs):
        # 1. 编码所有样本
        support_features = self.encoder(support_set, **kwargs)  # [N_support, D]
        query_features = self.encoder(query_set, **kwargs)      # [N_query, D]
        
        # 2. 计算类原型 (每个类的特征均值)
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # 3. 计算查询样本到各原型的距离
        distances = self._compute_distances(query_features, prototypes)
        
        # 4. 距离转概率 (距离越小，概率越大)
        log_probs = F.log_softmax(-distances, dim=-1)
        
        return {'log_probs': log_probs, 'prototypes': prototypes}
```

### 4.3 Episode采样

```python
class EpisodeSampler:
    """N-way K-shot Episode 采样器"""
    
    def __init__(self, n_way=2, k_shot=10, n_query=15):
        self.n_way = n_way      # 类别数 (human vs bot)
        self.k_shot = k_shot    # 每类支持样本数
        self.n_query = n_query  # 每类查询样本数
    
    def sample(self, dataset, indices):
        """
        从indices中采样一个episode
        返回: support_set, query_set
        """
        # 按类别分组
        class_indices = self._group_by_class(dataset, indices)
        
        # 采样support set (每类k_shot个)
        support_indices = []
        for c in range(self.n_way):
            support_indices.extend(random.sample(class_indices[c], self.k_shot))
        
        # 采样query set (不与support重叠)
        query_indices = []
        for c in range(self.n_way):
            remaining = [i for i in class_indices[c] if i not in support_indices]
            query_indices.extend(random.sample(remaining, self.n_query))
        
        return self._build_batch(dataset, support_indices, query_indices)
```

### 4.4 多模态编码器

```python
class MultiModalEncoder(nn.Module):
    """支持可配置模态组合的多模态编码器"""
    
    def __init__(self, config):
        # 根据配置初始化各编码器
        self.enabled_modalities = config.get('enabled_modalities', ['num', 'cat'])
        
        if 'num' in self.enabled_modalities:
            self.numerical_encoder = NumericalEncoder(8, 32, 64)
        if 'cat' in self.enabled_modalities:
            self.categorical_encoder = CategoricalEncoder([2,2,2,2,2], 16, 32)
        if 'text' in self.enabled_modalities:
            self.text_encoder = TextEncoder('xlm-roberta-base', 256)
        if 'graph' in self.enabled_modalities:
            self.graph_encoder = GraphEncoder(256, 128, 128)
        
        # 融合模块
        self.fusion = AttentionFusion(...)
    
    def forward(self, batch, texts=None, edge_index=None, **kwargs):
        embeddings = {}
        
        if self.numerical_encoder:
            embeddings['num'] = self.numerical_encoder(batch['num_features'])
        if self.categorical_encoder:
            embeddings['cat'] = self.categorical_encoder(batch['cat_features'])
        if self.text_encoder and texts:
            embeddings['text'] = self.text_encoder(texts)
        if self.graph_encoder and edge_index:
            embeddings['graph'] = self.graph_encoder(...)
        
        return self.fusion(**embeddings)
```

---

## 5. 训练流程

### 5.1 元训练循环

```python
class MetaTrainer:
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            # 训练阶段 (100 episodes)
            train_metrics = self.train_epoch(n_episodes=100)
            
            # 验证阶段 (50 episodes)
            val_metrics = self.validate(n_episodes=50)
            
            # 检查改进
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.patience:
                break
```

### 5.2 损失函数

```
Loss = NLLLoss(log_probs, true_labels)
     = -Σ log P(y_true | query)

目标: 最小化 query 样本到正确原型的距离
```

---

## 6. 实验设计

### 6.1 实验一：源域内验证

**目的**：验证原型网络在单一数据集上的有效性

**设置**：
- 数据集：Twibot-20
- 训练：train split上采样episode
- 验证：val split上评估
- 测试：test split上报告最终结果

### 6.2 实验二：跨域迁移（核心实验）

**目的**：验证跨平台少样本适应能力

**设置**：
- 元训练：Twibot-20全部数据
- 适应：Misbot的K个标注样本（K=1,5,10,20）
- 测试：Misbot测试集

**评估指标**：
- Accuracy, Precision, Recall, F1
- 不同K值下的性能曲线

### 6.3 实验三：消融实验

**目的**：分析各组件贡献

| 配置 | 模态 | 验证问题 |
|------|------|----------|
| ablation_num_cat | num + cat | 基线效果 |
| ablation_num_cat_text | num + cat + text | 文本的贡献 |
| ablation_all | num + cat + text + graph | 完整模型效果 |

---

## 7. 实施进度

### 第一阶段：数据准备 ✅
- [x] 统一预处理器实现
- [x] 流式加载优化
- [x] 运行预处理，验证输出

### 第二阶段：基础模型 ✅
- [x] 数值/分类编码器
- [x] 原型网络核心
- [x] Episode采样器
- [x] 元训练器
- [x] 评估器

### 第三阶段：完整模型 ✅
- [x] 文本编码器（XLM-RoBERTa）
- [x] 图编码器（GAT）
- [x] 注意力融合模块
- [x] 多模态训练支持

### 第四阶段：实验与论文
- [ ] 完整消融实验
- [ ] 结果分析
- [ ] 论文撰写

---

## 8. 预期贡献

1. **方法贡献**：首次将原型网络应用于跨平台社交机器人检测
2. **实验贡献**：建立Twibot-20→Misbot跨域评估基准
3. **实用价值**：提供少样本快速部署方案

---

## 9. 运行命令

```bash
# 数据预处理
python preprocess_unified.py --dataset all

# 训练 (源域: Twibot-20)
python experiments/train_source.py --config configs/default.yaml

# 评估 (目标域: Misbot)
python experiments/evaluate_target.py --model-path results/{timestamp}/best_model.pt

# 消融实验
python experiments/train_source.py --config configs/ablation_num_cat.yaml
python experiments/train_source.py --config configs/ablation_num_cat_text.yaml
python experiments/train_source.py --config configs/ablation_all.yaml
```
