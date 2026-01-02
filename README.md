# 跨平台社交机器人检测 - 原型网络模型

基于原型网络的跨平台社交机器人检测系统，采用元学习方法实现少样本跨域迁移。

## 项目概述

本项目实现了一个基于原型网络 (Prototypical Network) 的社交机器人检测系统。通过在源域 (Twibot-20) 上进行元训练，模型学习从少量样本中提取有效特征并构建类原型的能力，然后在目标域 (Misbot) 上仅用 5-10 个标注样本即可实现快速适应。

核心创新：使用度量学习替代固定分类头，使模型能够通过计算样本到类原型的距离进行分类，天然支持少样本跨域迁移。

## 目录结构

```
.
├── configs/              # 配置文件
│   ├── default.yaml     # 默认配置 (num + cat)
│   ├── ablation_num_cat.yaml      # 消融: 基线
│   ├── ablation_num_cat_text.yaml # 消融: +文本
│   └── ablation_all.yaml          # 消融: 全模态
├── dataset/              # 原始数据集
│   ├── Twibot-20/       # 源域数据
│   └── Misbot/          # 目标域数据
├── experiments/          # 实验脚本
│   ├── train_source.py  # 源域训练
│   └── evaluate_target.py # 跨域评估
├── processed_data/       # 预处理数据
│   ├── twibot20/        # 处理后的源域数据
│   └── misbot/          # 处理后的目标域数据
├── results/              # 实验结果
├── src/                  # 源代码
│   ├── config/          # 配置管理
│   ├── data/            # 数据加载与采样
│   ├── models/          # 模型定义
│   └── training/        # 训练与评估
├── tests/                # 测试代码
├── preprocess.py         # 数据预处理入口
├── preprocess_unified.py # 统一预处理系统
├── requirements.txt      # 依赖列表
├── DESIGN_DOCUMENT.md    # 设计文档
└── ALGORITHM.md          # 算法详解
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 预处理所有数据集
python preprocess_unified.py --dataset all

# 或单独预处理
python preprocess_unified.py --dataset twibot20
python preprocess_unified.py --dataset misbot
```

### 3. 预计算嵌入 (可选但推荐)

如果要使用文本或图模态，建议预先计算嵌入以大幅加速训练：

```bash
# 预计算文本嵌入
python precompute_text_embeddings.py --dataset all --device cuda

# 预计算图嵌入
python precompute_graph_embeddings.py --dataset all --device cuda
```

预计算后，训练时会自动使用预计算的嵌入，避免重复的 Transformer/RGCN 推理。

### 4. 训练模型

```bash
# 使用默认配置 (数值+分类特征)
python experiments/train_source.py --config configs/default.yaml

# 使用消融配置
python experiments/train_source.py --config configs/ablation_all.yaml

# 覆盖参数
python experiments/train_source.py --epochs 100 --lr 0.0005
```

### 5. 跨域评估

```bash
# 基本评估
python experiments/evaluate_target.py --model-path results/{timestamp}/best_model.pt

# 指定 K-shot 值
python experiments/evaluate_target.py --model-path results/{timestamp}/best_model.pt --k-shots 1 5 10 20

# 更多评估 episode
python experiments/evaluate_target.py --model-path results/{timestamp}/best_model.pt --n-episodes 200
```

## 模型架构

```
输入特征
    │
    ├── 数值特征 [8维] ──→ NumericalEncoder ──→ [64维]
    │   (followers, following, tweets, listed, age, ratio, username_len, desc_len)
    │
    ├── 分类特征 [5维] ──→ CategoricalEncoder ──→ [32维]
    │   (verified, protected, default_avatar, has_url, has_location)
    │
    ├── 文本特征 [可选] ──→ TextEncoder (XLM-RoBERTa) ──→ [256维]
    │   (description + tweets)
    │
    └── 图特征 [可选] ──→ GraphEncoder (RGCN) ──→ [128维]
        (社交网络结构，支持多关系类型)
                                                   │
                                                   ▼
                                        AttentionFusion / FusionModule
                                                   │
                                                   ▼
                                              [256维嵌入]
                                                   │
                                                   ▼
                                         PrototypicalNetwork
                                         (原型计算 + 距离分类)
                                                   │
                                                   ▼
                                              预测结果
```

## 核心特性

- **元学习训练**: 采用 Episode 训练方式 (N-way K-shot)，学习从少量样本泛化的能力
- **多模态融合**: 支持数值、分类、文本、图四种模态的灵活组合
- **注意力融合**: 使用注意力机制自动学习各模态的重要性权重
- **图特征编码**: 使用 RGCN (Relational GCN) 原生支持多关系类型
- **跨域迁移**: 在源域训练，目标域仅需少量样本即可适应
- **消融实验**: 预配置多种模态组合，便于分析各组件贡献

## 配置说明

主要配置参数 (`configs/default.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_way | 2 | Episode 类别数 (human vs bot) |
| k_shot | 10 | 每类 support 样本数 |
| n_query | 15 | 每类 query 样本数 |
| n_episodes_train | 100 | 每 epoch 训练 episode 数 |
| n_episodes_val | 50 | 每 epoch 验证 episode 数 |
| n_epochs | 200 | 最大训练轮数 |
| learning_rate | 0.001 | 主学习率 |
| text_learning_rate | 0.00001 | 文本编码器学习率 |
| patience | 10 | 早停耐心值 |
| distance_metric | euclidean | 距离度量 (euclidean/cosine) |
| enabled_modalities | ['num', 'cat'] | 启用的模态 |

## 消融实验配置

| 配置文件 | 启用模态 | 说明 |
|----------|----------|------|
| ablation_num_cat.yaml | num, cat | 基线模型 |
| ablation_num_cat_text.yaml | num, cat, text | +文本编码 |
| ablation_num_cat_graph.yaml | num, cat, graph | +图编码 (RGCN) |
| ablation_all.yaml | num, cat, text, graph | 完整模型 |

## 实验结果

在 Twibot-20 → Misbot 跨域迁移任务上的预期性能:

| K-shot | Accuracy | F1 Score |
|--------|----------|----------|
| 1-shot | ~62% | ~60% |
| 5-shot | ~72% | ~70% |
| 10-shot | ~75% | ~74% |
| 20-shot | ~78% | ~78% |

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 只运行单元测试
pytest tests/unit/ -v

# 只运行属性测试
pytest tests/property/ -v

# 查看测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- Transformers (XLM-RoBERTa)
- PyTorch Geometric 2.3+ (可选，用于图编码)
- Hypothesis (属性测试)
- pytest (测试框架)
- PyYAML (配置管理)

## 文档

- [DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md) - 研究设计与方法论
- [ALGORITHM.md](ALGORITHM.md) - 算法原理详解
- [src/README.md](src/README.md) - 源代码说明
- [experiments/README.md](experiments/README.md) - 实验脚本说明
- [configs/README.md](configs/README.md) - 配置说明
- [processed_data/README.md](processed_data/README.md) - 数据格式说明
- [dataset/README.md](dataset/README.md) - 原始数据说明
- [results/README.md](results/README.md) - 结果目录说明

## License

MIT License
