# src/training/ - 训练和评估模块

本目录包含模型训练和评估相关代码。

## 文件说明

### meta_trainer.py

`MetaTrainer` 类 - 元训练器，实现 episode 训练循环。

**主要方法:**
- `train_epoch(n_episodes)`: 训练一个 epoch，返回平均损失和准确率
- `validate(n_episodes)`: 在验证集上评估
- `train()`: 完整训练流程，包含早停和检查点保存

**训练流程:**
1. 每个 epoch 采样多个 episode
2. 对每个 episode:
   - 从 support set 计算类原型
   - 对 query set 计算预测概率
   - 计算负对数似然损失
3. 反向传播更新参数
4. 验证集评估
5. 保存最佳模型

**检查点保存:**
- `best_model.pt`: 验证损失最低的模型
- `checkpoint_epoch_N.pt`: 每个 epoch 的检查点

### evaluator.py

`Evaluator` 类 - 少样本评估器。

**主要方法:**
- `few_shot_evaluate(support_set, test_dataset, test_indices)`: 少样本评估
- `compute_metrics(predictions, labels)`: 计算评估指标

**评估指标:**
- `accuracy`: 准确率 = (TP + TN) / Total
- `precision`: 精确率 = TP / (TP + FP)
- `recall`: 召回率 = TP / (TP + FN)
- `f1`: F1 分数 = 2 × precision × recall / (precision + recall)

**评估流程:**
1. 冻结编码器参数 (eval 模式)
2. 从 support set 计算类原型
3. 对测试集所有样本计算预测
4. 计算并返回评估指标

## 使用示例

```python
from src.training import MetaTrainer, Evaluator

# 训练
trainer_config = {
    'n_way': 2,
    'k_shot': 5,
    'n_query': 15,
    'n_episodes_train': 100,
    'n_episodes_val': 50,
    'n_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 10,
    'output_dir': 'results/'
}
trainer = MetaTrainer(model, dataset, trainer_config)
history = trainer.train()

# 评估
evaluator = Evaluator(model)
metrics = evaluator.few_shot_evaluate(
    support_set,
    test_dataset,
    test_indices
)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## 训练配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| n_way | Episode 类别数 | 2 |
| k_shot | 每类 support 样本数 | 5 |
| n_query | 每类 query 样本数 | 15 |
| n_episodes_train | 每 epoch 训练 episode 数 | 100 |
| n_episodes_val | 验证 episode 数 | 50 |
| n_epochs | 最大训练轮数 | 100 |
| learning_rate | 学习率 | 1e-3 |
| weight_decay | 权重衰减 | 1e-4 |
| patience | 早停耐心值 | 10 |

## 早停机制

当验证损失连续 `patience` 个 epoch 没有改善时，训练自动停止。这有助于:
- 防止过拟合
- 节省训练时间
- 保留最佳模型
