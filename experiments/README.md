# experiments/ - 实验脚本目录

本目录包含训练和评估实验的入口脚本。

## 文件说明

### train_source.py

源域训练脚本 - 在 Twibot-20 上训练原型网络。

**功能:**
- 加载配置和数据集
- 初始化多模态编码器和原型网络
- 运行元训练循环
- 保存最佳模型和训练历史
- 支持消融实验配置

**命令行参数:**
```
--config, -c     配置文件路径 (默认: configs/default.yaml)
--dataset        数据集名称 (默认: twibot20)
--output-dir     输出目录 (覆盖配置)
--epochs         训练轮数 (覆盖配置)
--lr             学习率 (覆盖配置)
--seed           随机种子 (覆盖配置)
--device         设备 (cuda/cpu, 自动检测)
--modalities     启用的模态 (覆盖配置)
```

**使用示例:**
```bash
# 使用默认配置训练
python experiments/train_source.py

# 指定配置文件
python experiments/train_source.py --config configs/ablation_all.yaml

# 覆盖部分参数
python experiments/train_source.py --epochs 50 --lr 0.0005

# 指定输出目录
python experiments/train_source.py --output-dir results/exp1

# 指定模态
python experiments/train_source.py --modalities num cat text
```

**输出文件:**
- `best_model.pt`: 最佳模型检查点（包含模型权重、优化器状态、配置）
- `train.log`: 训练日志

### evaluate_target.py

跨域评估脚本 - 在 Misbot 上进行少样本评估。

**功能:**
- 加载预训练模型
- 在目标域进行少样本适应
- 支持不同 K-shot 值评估
- 保存评估结果

**命令行参数:**
```
--config, -c     配置文件路径 (默认: configs/default.yaml)
--model-path, -m 预训练模型路径 (必需)
--dataset        目标数据集 (默认: misbot)
--k-shots        K-shot 值列表 (默认: 1 5 10 20)
--n-episodes     评估 episode 数 (默认: 100)
--output-dir     输出目录
--device         设备 (cuda/cpu)
```

**使用示例:**
```bash
# 基本评估
python experiments/evaluate_target.py --model-path results/best_model.pt

# 指定 K-shot 值
python experiments/evaluate_target.py --model-path results/best_model.pt --k-shots 1 5 10

# 更多评估 episode
python experiments/evaluate_target.py --model-path results/best_model.pt --n-episodes 200

# 指定输出目录
python experiments/evaluate_target.py --model-path results/best_model.pt --output-dir results/eval
```

**输出文件:**
- `eval_misbot.json`: 评估结果 (各 K-shot 的指标)
- `eval_YYYYMMDD_HHMMSS.log`: 评估日志

## 完整实验流程

```bash
# 1. 数据预处理
python preprocess_unified.py --dataset all

# 2. 在源域 (Twibot-20) 训练
python experiments/train_source.py \
    --config configs/default.yaml \
    --output-dir results/baseline

# 3. 在目标域 (Misbot) 评估
python experiments/evaluate_target.py \
    --model-path results/baseline/best_model.pt \
    --output-dir results/baseline/eval
```

## 消融实验流程

```bash
# 基线 (num + cat)
python experiments/train_source.py --config configs/ablation_num_cat.yaml

# 加文本 (num + cat + text)
python experiments/train_source.py --config configs/ablation_num_cat_text.yaml

# 完整模型 (num + cat + text + graph)
python experiments/train_source.py --config configs/ablation_all.yaml
```

## 实验结果示例

训练输出:
```
Training: 200 epochs, 100 episodes/epoch
Enabled modalities: num, cat
Epoch   1/200 | Loss: 1.7404/0.7302 | Acc: 0.5760/0.5993 | F1: 0.5234/0.5456
Epoch   2/200 | Loss: 0.7048/0.6514 | Acc: 0.6293/0.6167 | F1: 0.6012/0.5989
...
Epoch  45/200 | Loss: 0.4523/0.5234 | Acc: 0.7823/0.7456 | F1: 0.7756/0.7389 *
...
Early stopping at epoch 55 (no improvement for 10 epochs)
Done. Best val loss: 0.5234
```

评估输出:
```
Loading model from results/baseline/best_model.pt
Evaluating on misbot with k_shots=[1, 5, 10, 20]

K-shot Evaluation Results:
  1-shot: Acc=0.6234, Prec=0.6123, Rec=0.6345, F1=0.6012
  5-shot: Acc=0.7156, Prec=0.7089, Rec=0.7223, F1=0.7023
 10-shot: Acc=0.7534, Prec=0.7456, Rec=0.7612, F1=0.7412
 20-shot: Acc=0.7823, Prec=0.7789, Rec=0.7856, F1=0.7756

Results saved to results/baseline/eval/eval_misbot.json
```

## 检查点格式

`best_model.pt` 包含:
```python
{
    'epoch': int,                    # 训练轮数
    'model_state_dict': dict,        # 模型权重
    'optimizer_state_dict': dict,    # 优化器状态
    'best_val_loss': float,          # 最佳验证损失
    'config': dict,                  # 训练配置
    'enabled_modalities': list       # 启用的模态
}
```

## 评估结果格式

`eval_misbot.json`:
```json
{
    "1": {
        "accuracy": 0.6234,
        "precision": 0.6123,
        "recall": 0.6345,
        "f1": 0.6012
    },
    "5": {
        "accuracy": 0.7156,
        "precision": 0.7089,
        "recall": 0.7223,
        "f1": 0.7023
    },
    "10": {...},
    "20": {...}
}
```
