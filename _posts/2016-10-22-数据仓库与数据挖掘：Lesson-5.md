---
layout: post
title: 数据仓库与数据挖掘：Lesson 5
tags: [数据仓库与数据挖掘]
comments: true
share: true
---

## Clustering

### Partition method

Construct a partition of a database D of n objects into a set of k clusters s.t. minimize sum of squared distance

- K-mean
  - 迭代计算中心
  - 如何初始中心是个关键问题
    - 随机选择
    - 基于其他聚类算法[效果不一定好，但是效率高]的结果估算中心
  - 优势
    - 可扩展性强
    - 效率较高
    - 可实现局部最优 [退火算法和遗传算法等可以用找全局最优]
  - 缺点
    - 类数必须事先确定
    - 对噪音数据处理不好
    - 某些特殊分布无法划分（如凹型的）
- K-medoids [PAM]

有**噪音**和**奇异点**时，PAM比 k-means 鲁棒

### Density based method

#### DBSCAN

`Definition`

- $\varepsilon-neigborbood$
- core object
- direct density reachable
- density reachable [use chain build by core-object]
- density connected

Base on above definition we can get the **basic idea** for DBSCAN

- if an object p is density connected to q , then q and p belong to the same cluster

优点：可以区分任意形状的，并能较好处理噪音，不需要事先给定类别

缺点：对于密度可变的问题表现不好（OPTICS对该问题的改进），链条现象（CURE对该问题改进）

#### OPTICS

核心思想：

​	为每个数据对象计算一个顺序值，这些顺序值代表了数据对象基于密度的族结构

`Definition`

- core distance (给定$MinPts$，使某个对象称为核心对象的最小距离)

- reachable-distance   $Max\{O的核心距离，O与P的欧几里得距离\}$

  ​



### 基于层次的聚类

- 族间距离



### 基于模型的聚类

#### GMM

- 基本假设：数据服从高斯混合分布
- 逐步迭代类似K-means

### EM



