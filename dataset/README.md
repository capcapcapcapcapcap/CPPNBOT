# dataset/ - 原始数据集目录

本目录包含原始的社交机器人检测数据集。

## 目录结构

```
dataset/
├── README.md
├── Twibot-20/          # Twitter 机器人检测数据集 (源域)
│   ├── node.json       # 用户节点信息
│   ├── edge.json       # 用户关系边
│   ├── label.json      # 用户标签
│   ├── split.json      # 数据划分
│   └── ...
│
└── Misbot/             # 跨平台机器人检测数据集 (目标域)
    ├── node.json
    ├── edge.json
    ├── label.json
    ├── split.json
    └── ...
```

## 数据集说明

### Twibot-20 (源域)

Twitter 机器人检测基准数据集，包含大规模的 Twitter 用户数据。

**数据规模:**
- 用户数: 229,580
- 标注用户: 11,826
- 边数: 关注关系

**用于:** 源域训练，学习通用的机器人检测模式

**特点:**
- 英文文本
- 完整的用户属性
- 丰富的社交网络结构

### Misbot (目标域)

跨平台社交机器人检测数据集。

**用于:** 目标域评估，测试少样本跨域迁移能力

**特点:**
- 中文文本
- 不同的平台特性
- 用于验证跨语言、跨平台泛化能力

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
        "protected": false,
        "default_profile": false,
        "default_profile_image": false,
        "description": "用户简介",
        "created_at": "创建时间",
        "url": "用户URL",
        "location": "位置"
    }
}
```

### edge.json

用户关系边，表示社交网络结构。

```json
[
    {"source": "user_id_1", "target": "user_id_2", "relation": "follow"},
    {"source": "user_id_2", "target": "user_id_3", "relation": "friend"},
    ...
]
```

### label.json

用户标签。

```json
{
    "user_id_1": "bot",
    "user_id_2": "human",
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
# 预处理所有数据集
python preprocess_unified.py --dataset all

# 只预处理 Twibot-20
python preprocess_unified.py --dataset twibot20

# 只预处理 Misbot
python preprocess_unified.py --dataset misbot
```

预处理后的数据保存在 `processed_data/` 目录。

## 特征提取

预处理脚本从原始数据中提取以下特征:

**数值特征 (8维):**
1. followers_count - 粉丝数
2. following_count - 关注数
3. tweet_count - 推文数
4. listed_count - 被列表收录次数
5. account_age_days - 账户年龄天数
6. followers_following_ratio - 粉丝/关注比
7. username_length - 用户名长度
8. description_length - 简介长度

**分类特征 (5维):**
1. verified - 是否验证
2. protected - 是否受保护
3. default_avatar - 是否默认头像
4. has_url - 是否有URL
5. has_location - 是否有位置

**文本特征:**
- description - 用户简介
- tweets - 用户推文列表

**图结构:**
- edge_index - 边索引
- edge_type - 边类型

## 数据引用

如果使用这些数据集，请引用原始论文:

**Twibot-20:**
```bibtex
@article{Feng2021TwiBot20AC,
  title={TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark},
  author={Shangbin Feng and Herun Wan and Ningnan Wang and Jundong Li and Minnan Luo},
  journal={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  year={2021}
}
```
**Misbot:**
```bibtex
@inproceedings{wan-etal-2025-social,
    title = "How Do Social Bots Participate in Misinformation Spread? A Comprehensive Dataset and Analysis",
    author = "Wan, Herun  and
      Luo, Minnan  and
      Ma, Zihan  and
      Dai, Guang  and
      Zhao, Xiang",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1604/",
    pages = "31481--31504",
    ISBN = "979-8-89176-332-6"
}
}
```
## 注意事项

1. 原始数据集文件较大，不建议直接用于训练
2. 请先运行预处理脚本生成 `processed_data/` 中的数据
3. 预处理会进行特征标准化、标签编码等操作
4. 流式加载优化可处理大文件内存问题
