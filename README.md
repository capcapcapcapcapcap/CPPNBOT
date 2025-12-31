# 跨平台社交机器人检测 - 原型网络模型

基于原型网络的跨平台社交机器人检测模型，采用元学习方法实现少样本跨域迁移。

## 项目概述

本项目实现了一个基于原型网络 (Prototypical Network) 的社交机器人检测系统。通过在源域 (Twibot-20) 上进行元训练，模型学习从少量样本中提取有效特征并构建类原型的能力，然后在目标域 (Misbot) 上仅用 5-10 个标注样本即可实现快速适应。

## 目录结构

```
.
├── configs/              # 配置文件
├── dataset/              # 原始数据集
├── experiments/          # 实验脚本
├── processed_data/       # 预处理数据
├── results/              # 实验结果
├── src/                  # 源代码
│   ├── config/          # 配置管理
│   ├── data/            # 数据加载
│   ├── models/          # 模型定义
│   └── training/        # 训练评估
├── tests/                # 测试代码
│   ├── unit/            # 单元测试
│   ├── property/        # 属性测试
│   └── integration/     # 集成测试
├── preprocess.py         # 数据预处理脚本
├── requirements.txt      # 依赖列表
└── DESIGN_DOCUMENT.md    # 设计文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
python preprocess.py
```

### 3. 训练模型

```bash
python experiments/train_source.py --config configs/default.yaml
```

### 4. 跨域评估

```bash
python experiments/evaluate_target.py --model results/best_model.pt
```

## 模型架构

```
输入特征
    │
    ├── 数值特征 [5维] ──→ NumericalEncoder ──→ [64维]
    │                                              │
    └── 分类特征 [3维] ──→ CategoricalEncoder ──→ [32维]
                                                   │
                                                   ▼
                                            FusionModule
                                                   │
                                                   ▼
                                              [256维嵌入]
                                                   │
                                                   ▼
                                         PrototypicalNetwork
                                                   │
                                                   ▼
                                              预测结果
```

## 核心特性

- **元学习**: 采用 Episode 训练方式，学习从少量样本泛化的能力
- **多模态融合**: 结合数值特征和分类特征
- **跨域迁移**: 在源域训练，目标域少样本适应
- **属性测试**: 使用 Hypothesis 框架验证模型正确性

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 只运行属性测试
pytest tests/property/ -v

# 查看测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 配置说明

主要配置参数 (`configs/default.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_way | 2 | Episode 类别数 |
| k_shot | 5 | 每类 support 样本数 |
| n_query | 15 | 每类 query 样本数 |
| n_epochs | 100 | 最大训练轮数 |
| learning_rate | 1e-3 | 学习率 |
| distance_metric | euclidean | 距离度量 |

## 实验结果

在 Twibot-20 → Misbot 跨域迁移任务上:

| K-shot | Accuracy | F1 Score |
|--------|----------|----------|
| 1-shot | ~62% | ~60% |
| 5-shot | ~72% | ~70% |
| 10-shot | ~75% | ~74% |
| 20-shot | ~78% | ~78% |

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- Hypothesis (属性测试)
- pytest (测试框架)
- PyYAML (配置管理)

## 文档

每个目录下都有详细的 README.md 文档:

- [src/README.md](src/README.md) - 源代码说明
- [tests/README.md](tests/README.md) - 测试说明
- [experiments/README.md](experiments/README.md) - 实验脚本说明
- [configs/README.md](configs/README.md) - 配置说明
- [processed_data/README.md](processed_data/README.md) - 数据格式说明

## License

MIT License
