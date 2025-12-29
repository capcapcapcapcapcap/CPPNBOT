# 社交机器人检测数据集集合

## 概述

本目录包含两个主要的社交媒体机器人检测数据集，支持传统机器学习和图神经网络方法的研究。

## 数据集对比

| 特性 | Misbot | Twibot-20 |
|------|--------|-----------|
| **语言** | 中文微博 | 英文Twitter |
| **用户数** | 99,874 | 224,818 |
| **标注用户** | 99,874 | 11,826 |
| **推文数** | 2.4M | 33.5M |
| **数据来源** | 微博平台 | Twitter API |
| **图结构** | 支持 | 原生支持 |
| **文件大小** | 401MB | 7.5GB |

## 数据集目录

### 1.Misbot (中文微博数据集)
```
Misbot_Graph/
├── node.json          # 节点数据 (451MB)
├── edge.json          # 边关系 (73MB)
├── label.json         # 标签数据 (2MB)
├── split.json         # 数据划分 (2MB)
├── sample.json        # 样本数据 (1MB)
└── README.md          # 说明文档
```

**特点**：
- 异构图结构：用户节点 + 微博节点
- 4种关系类型：post, mention, retweet, follow
- 与Twibot-20格式完全兼容
- 支持图神经网络建模

### 3. Twibot-20 (英文Twitter数据集)
```
Twibot-20/
├── node.json          # 节点数据 (6.6GB)
├── edge.json          # 边关系 (734MB)
├── label.json         # 标签数据 (0.3MB)
├── split.json         # 数据划分 (0.3MB)
├── sample.json        # 样本数据 (3MB)
└── README.md          # 说明文档
```

**特点**：
- 大规模Twitter社交网络图
- 真实API数据，38个用户属性
- 3种关系类型：post, friend, follow
- 支持大规模图神经网络研究

## 格式兼容性

两个图结构数据集采用相同的边格式：
```json
// 统一格式: [关系类型, 目标ID]
["post", "t_xxx"]      // 发布关系
["mention", "u_xxx"]   // 提及关系  
["follow", "u_xxx"]    // 关注关系
```

这使得可以使用相同的代码处理两个数据集。

