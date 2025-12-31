# dataset/ - 原始数据集目录

本目录包含原始的社交机器人检测数据集。

## 目录结构

```
dataset/
├── README.md
├── Twibot-20/          # Twitter 机器人检测数据集
│   ├── node.json       # 用户节点信息
│   ├── edge.json       # 用户关系边
│   ├── label.json      # 用户标签
│   ├── split.json      # 数据划分
│   ├── sample.json     # 采样信息
│   └── README.md       # 数据集说明
│
└── Misbot/             # 跨平台机器人检测数据集
    ├── node.json
    ├── edge.json
    ├── label.json
    ├── split.json
    ├── sample.json
    └── README.md
```

## 数据集说明

### Twibot-20 (源域)

Twitter 机器人检测基准数据集，包含大规模的 Twitter 用户数据。

**数据规模:**
- 用户数: 229,580
- 标注用户: 11,826
- 边数: (关注关系)

**用于:** 源域训练，学习通用的机器人检测模式

### Misbot (目标域)

跨平台社交机器人检测数据集。

**用于:** 目标域评估，测试少样本跨域迁移能力

## 文件格式

### node.json
用户节点信息，包含用户属性特征。

```json
{
    "user_id": {
        "id": "user_id",
        "name": "用户名",
        "screen_name": "显示名",
        "followers_count": 1000,
        "friends_count": 500,
        "statuses_count": 2000,
        "favourites_count": 300,
        "listed_count": 10,
        "verified": false,
        "default_profile": false,
        "default_profile_image": false,
        "description": "用户简介",
        "created_at": "创建时间"
    }
}
```

### edge.json
用户关系边，表示关注关系。

```json
[
    {"source": "user_id_1", "target": "user_id_2", "relation": "follow"},
    ...
]
```

### label.json
用户标签。

```json
{
    "user_id": "bot",    // 或 "human"
    ...
}
```

### split.json
数据划分信息。

```json
{
    "train": ["user_id_1", "user_id_2", ...],
    "val": ["user_id_3", "user_id_4", ...],
    "test": ["user_id_5", "user_id_6", ...]
}
```

## 数据预处理

使用预处理脚本将原始数据转换为模型可用的格式:

```bash
# 预处理 Twibot-20
python preprocess.py --input dataset/Twibot-20 --output processed_data/twibot20

# 预处理 Misbot
python preprocess.py --input dataset/Misbot --output processed_data/misbot

# 或使用统一预处理脚本
python preprocess_unified.py
```

## 数据引用

如果使用这些数据集，请引用原始论文:

**Twibot-20:**
```bibtex
@inproceedings{feng2021twibot,
    title={TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark},
    author={...},
    booktitle={...},
    year={2021}
}
```

## 注意事项

1. 原始数据集较大，不建议直接用于训练
2. 请先运行预处理脚本生成 `processed_data/` 中的数据
3. 预处理会进行特征归一化、标签编码等操作
4. 图结构数据用于未来的图神经网络扩展
