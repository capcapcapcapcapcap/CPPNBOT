# src/training/ - 训练和评估模块

本目录包含模型训练和评估相关代码。

## 文件说明

### meta_trainer.py

`MetaTrainer` 类 - 元训练器，实现 episode 训练循环。

**主要方法:**
- `train_epoch(n_episodes)`: 训练一个 epoch，返回平均损失和指标
- `validate(n_episodes)`: 在验证集上评估
- `train()`: 完整训练流程，包含早停和检查点保存
- `save_checkpoint(is_best)`: 保存模型检查点
- `load_checkpoint(filepath)`: 加载模型检查点

**多模态支持:**
- 自动加载文本数据 (如果启用 text 模态)
- 自动加载图数据 (如果启用 graph 模态)
- 分离学习率 (文本编码器使用更小的学习率)
- 可选冻结文本骨干网络

**训练流程:**
1. 每个 epoch 采样多个 episode
2. 对每个 episode:
   - 从 support set 计算类原型
   - 对 query set 计算预测概率
   - 计算负对数似然损失
3. 反向传播更新参数
4. 验证集评估
5. 保存最佳模型

**检查点内容:**
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'best_val_loss': float,
    'config': dict,
    'enabled_modalities': list
}
```

### evaluator.py

`Evaluator` 类 - 少样本评估器。

**主要方法:**
- `few_shot_evaluate(support_set, test_dataset, test_indices)`: 少样本评估
- `evaluate_with_k_shot(dataset, train_idx, test_idx, k_shot, n_episodes)`: 指定 K-shot 评估
- `evaluate_multiple_k_shots(dataset, train_idx, test_idx, k_shots, n_episodes)`: 多 K-shot 评估
- `set_dataset_data(dataset)`: 加载文本和图数据

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

# 训练配置
trainer_config = {
    'n_way': 2,
    'k_shot': 10,
    'n_query': 15,
    'n_episodes_train': 100,
    'n_episodes_val': 50,
    'n_epochs': 200,
    'learning_rate': 0.001,
    'text_learning_rate': 0.00001,
    'weight_decay': 0.0001,
    'patience': 10,
    'output_dir': 'results/',
    'enabled_modalities': ['num', 'cat'],
    'text_freeze_backbone': True
}

# 训练
trainer = MetaTrainer(model, dataset, trainer_config)
history = trainer.train()

# 评估
evaluator = Evaluator(model, enabled_modalities=['num', 'cat'])
results = evaluator.evaluate_multiple_k_shots(
    dataset,
    train_indices,
    test_indices,
    k_shots=[1, 5, 10, 20],
    n_episodes=100
)

for k, metrics in results.items():
    print(f"{k}-shot: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
```

## 训练配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| n_way | Episode 类别数 | 2 |
| k_shot | 每类 support 样本数 | 10 |
| n_query | 每类 query 样本数 | 15 |
| n_episodes_train | 每 epoch 训练 episode 数 | 100 |
| n_episodes_val | 验证 episode 数 | 50 |
| n_epochs | 最大训练轮数 | 200 |
| learning_rate | 主学习率 | 0.001 |
| text_learning_rate | 文本编码器学习率 | 0.00001 |
| weight_decay | 权重衰减 | 0.0001 |
| patience | 早停耐心值 | 10 |
| enabled_modalities | 启用的模态 | ['num', 'cat'] |
| text_freeze_backbone | 冻结文本骨干 | True |

## 早停机制

当验证损失连续 `patience` 个 epoch 没有改善时，训练自动停止。这有助于:
- 防止过拟合
- 节省训练时间
- 保留最佳模型

## 日志输出格式

```
Training: 200 epochs, 100 episodes/epoch
Enabled modalities: num, cat
Epoch   1/200 | Loss: 1.7404/0.7302 | Acc: 0.5760/0.5993 | F1: 0.5234/0.5456
Epoch   2/200 | Loss: 0.7048/0.6514 | Acc: 0.6293/0.6167 | F1: 0.6012/0.5989 *
...
Early stopping at epoch 55 (no improvement for 10 epochs)
Done. Best val loss: 0.5234
```

`*` 标记表示该 epoch 验证损失有改善，模型已保存。
