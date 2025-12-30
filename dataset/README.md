# 数据集说明

本项目使用两个社交机器人检测数据集：

## Twibot-20 (英文Twitter)

| 文件 | 大小 | 说明 |
|------|------|------|
| node.json | 6.62 GB | 用户和推文节点 |
| edge.json | 734 MB | 社交关系 |
| label.json | 0.32 MB | 用户标签 |
| split.json | 0.28 MB | 数据划分 |

**统计**：229,580用户 + 33,488,192推文，11,826个标注用户

**格式**：
- node.json: `{node_id: {属性...}, ...}` 字典格式
- 用户ID以`u`开头，推文ID以`t`开头
- 关系类型: post, friend, follow

## Misbot (中文微博)

| 文件 | 大小 | 说明 |
|------|------|------|
| node.json | 451 MB | 用户和推文节点 |
| edge.json | 73 MB | 社交关系 |
| label.json | 2.3 MB | 用户标签 |
| split.json | 1.6 MB | 数据划分 |

**统计**：99,874用户 + 2,427,195推文，全部用户已标注

**格式**：
- node.json: `{node_id: {属性...}, ...}` 字典格式
- 用户ID格式: `train_u{数字}`
- 推文ID格式: `t_train_u{数字}_{序号}`
- 关系类型: post, mention, retweet, follow

## 数据结构对比

| 特征 | Twibot-20 | Misbot |
|------|-----------|--------|
| 语言 | 英文 | 中文 |
| 数值特征 | followers, following, listed, tweets | followers, following, tweets |
| 分类特征 | verified, protected, default_avatar | 20维categorical |
| 文本 | description + tweets | description + tweets |

详细格式见各数据集目录下的 `sample.json`。
