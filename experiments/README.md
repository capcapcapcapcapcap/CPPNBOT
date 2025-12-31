# experiments/ - 实验脚本目录

本目录包含训练和评估实验的入口脚本。

## 文件说明

### train_source.py

源域训练脚本 - 在 Twibot-20 上训练原型网络。

**功能:**
- 加载配置和数据集
- 初始化模型和训练器
- 运行元训练循环
- 保存最佳模型和训练历史
- 记录训练日志

**命令行参数:**
```
--config, -c     配置文件路径 (默认: configs/default.yaml)
--dataset        数据集名称 (默认: twibot20)
--output-dir     输出目录 (覆盖配置)
--epochs         训练轮数 (覆盖配置)
--lr             学习率 (覆盖配置)
--seed           随机种子 (覆盖配置)
--device         设备 (cuda/cpu, 自动检测)
```

**使用示例:**
```bash
# 使用默认配置训练
python experiments/train_source.py

# 指定配置文件
python experiments/train_source.py --config configs/custom.yaml

# 覆盖部分参数
python experiments/train_source.py --epochs 50 --lr 0.0005

# 指定输出目录
python experiments/train_source.py --output-dir results/exp1
```

**输出文件:**
- `best_model.pt`: 最佳模型检查点
- `checkpoint_epoch_N.pt`: 每个 epoch 的检查点
- `training_history.json`: 训练历史 (损失、准确率)
- `train_YYYYMMDD_HHMMSS.log`: 训练日志

### evaluate_target.py

跨域评估脚本 - 在 Misbot 上进行少样本评估。

**功能:**
- 加载预训练模型
- 在目标域进行少样本适应
- 支持不同 K-shot 值评估
- 保存评估结果

**命令行参数:**
```
--model          预训练模型路径 (必需)
--config, -c     配置文件路径 (默认: configs/default.yaml)
--dataset        目标数据集 (默认: misbot)
--k-shots        K-shot 值列表 (默认: 1,5,10,20)
--n-episodes     评估 episode 数 (默认: 100)
--output-dir     输出目录
--device         设备 (cuda/cpu)
```

**使用示例:**
```bash
# 基本评估
python experiments/evaluate_target.py --model results/best_model.pt

# 指定 K-shot 值
python experiments/evaluate_target.py --model results/best_model.pt --k-shots 1,5,10

# 更多评估 episode
python experiments/evaluate_target.py --model results/best_model.pt --n-episodes 200
```

**输出文件:**
- `evaluation_results.json`: 评估结果 (各 K-shot 的指标)
- `evaluate_YYYYMMDD_HHMMSS.log`: 评估日志

## 完整实验流程

```bash
# 1. 在源域 (Twibot-20) 训练
python experiments/train_source.py \
    --config configs/default.yaml \
    --output-dir results/exp1

# 2. 在目标域 (Misbot) 评估
python experiments/evaluate_target.py \
    --model results/exp1/best_model.pt \
    --output-dir results/exp1/eval
```

## 实验结果示例

训练输出:
```
Epoch 1/100 - Train Loss: 1.7404, Train Acc: 0.5760, Val Loss: 0.7302, Val Acc: 0.5993
Epoch 2/100 - Train Loss: 0.7048, Train Acc: 0.6293, Val Loss: 0.6514, Val Acc: 0.6167
...
Training completed!
Best validation loss: 0.5234
```

评估输出:
```
K-shot Evaluation Results on Misbot:
  1-shot: Acc=0.6234, F1=0.6012
  5-shot: Acc=0.7156, F1=0.7023
 10-shot: Acc=0.7534, F1=0.7412
 20-shot: Acc=0.7823, F1=0.7756
```
