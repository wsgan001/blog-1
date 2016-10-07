---
layout: post
title: 统计机器学习：Introduction
tags: [统计机器学习]
---


> 机器学习是一门**人工智能**的科学，该领域的主要研究对象是人工智能，特别是在**经验**学习中改善具体算法的性能

机器学习核心功能：让分析算法无需人类干预的显示程式即可对最新数据进行学习

### 机器学习的发展

- 神经科学，认知科学
- 数学
- 计算

### 机器学习

学习的目的在于使期望风险最小化，但可利用信息只有样本，期望风险往往无法计算  **[经验风险最小化归纳原则]**

### 机器学习三类基本问题

#### 分类

$$
L(f(X,w),y)  =
\begin{cases}
1, & f(X,w) \neq y \cr
0 , & f(X,w) = y \cr
\end{cases}
$$

#### 回归

$$
L(f(X,w),y) = (f(X,w)-y )^2
$$

#### 概率密度估计

$$
L(p(X,w)) = -log(p(X,w))
$$



### 学习分类

#### 有监督学习

`标定的训练数据`

典型方法

- 全局：BN, NN,SVM, Decision Tree 
- 局部：KNN、CBR(Case-base reasoning)

> Case-based reasoning (CBR), broadly construed, is the process of solving new problems based on the solutions of similar past problems. An auto mechanic who fixes an engine by recalling another car that exhibited similar symptoms is using case-based reasoning.

#### 无监督学习

`不存在标定的训练数据`

典型方法

- K-means、SOM….

#### 半监督学习

`结合（少量的）标定训练数据和（大量的）未标定数据来进行学习`

典型方法

- Co-training、EM、Latent variables….

#### 增强学习(Reinforcement Learning)

外部环境对输出只给出**评价信息**而非正确答案，学习机通过**强化受奖励的动作**来改善自身的性能。

#### 多任务学习

Learns a problem together with other related problems at the same time, using a shared representation.

### 学习模型

#### 单学习模型
- Linear models
- Kernel methods
- Neural networks 
- Probabilistic models 
- Decision trees

#### 组合模型

- Boosting
  - 结合低性能学习模型来产生一个强大的分类器组
- Bagging
  - 结合多个不稳定学习模型来产生稳定预测
- 主动学习（Active Learning）

> [**What's the similarities and differences beetween this 3 methods: bagging, boosting, stacking?**](http://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning)
>
> All three are so-called "meta-algorithms": approaches to combine several machine learning techniques into one predictive model in order to decrease the variable (bagging), bias (boosting) or improving the predictive force (stacking alias ensemble).
>
> Every algorithm consists of two steps:
>
> Producing a distribution of simple ML models on subsets of the original data.
> Combining the distribution into one "aggregated" model.
> Here is a short description of all three methods:
>
> **[Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)** (stands for Bootstrap Aggregation) is the way decrease the variance of your prediction by generating additional data for training from your original dataset using combinations with repetitions to produce multisets of the same cardinality/size as your original data. By increasing the size of your training set you can't improve the model predictive force, but just decrease the variance, narrowly tuning the prediction to expected outcome.[Each predictor in ensemble is created by taking a bootstrap sample of the data]
>
>
> **[Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning))** is a two-step approach, where one first uses subsets of the original data to produce a series of averagely performing models and then "boosts" their performance by combining them together using a particular cost function (=majority vote). Unlike bagging, in the classical boosting the subset creation is not random and depends upon the performance of the previous models: every new subsets contains the elements that were (likely to be) mis-classified by previous models.
>
>
> **Stacking** is a similar to boosting: you also apply several models to you original data. The difference here is, however, that you don't have just an empirical formula for your weight function, rather you introduce a meta-level and use another model/approach to estimate the input together with outputs of every model to estimate the weights or, in other words, to determine what models perform well and what badly given these input data.
>
> ![](http://ww2.sinaimg.cn/large/7853084cjw1f81ewgf9soj20np06wt9o.jpg)



