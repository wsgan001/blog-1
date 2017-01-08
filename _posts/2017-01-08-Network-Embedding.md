---
layout: post
title: data science winter school note - Network Embedding
date: 2017-01-08
tags: [data science winter school note]
comments: true
share: true
---


### Concept

- Representation Learning

- Distributed Representation

- Embedding

- Softmax

  > It transforms a K-dimensional real vector into a **probability distribution**

### Word2Vec

#### Architecture

1. CBOW

   predicts the current word using **surrounding contexts** 

2. Skip-Gram
   predicts surrounding words using the **current word**


![](https://ww1.sinaimg.cn/large/006tNbRwjw1fbj0hjnvobj31ck0a4n06.jpg)

#### Solver

1. Hierarchical Softmax
2. Negative Sampling

### Doc2Vec

#### Architecture

![](https://ww4.sinaimg.cn/large/006tNbRwjw1fbizao8yhgj31740budh5.jpg)



>  From language mdeling to graph
>
>  Node -> Word , Path -> Sentence

### DeepWalk

Get a node ids sequence then run **word2vec**

How to get the ids sequence ?    Generate a **random paths** !

Direction choose:

![](https://ww2.sinaimg.cn/large/006tNbRwjw1fbizwor855j31920aajte.jpg)

### Node2Vec

Most is same as DeepWalk ,  but choose another way for generating a random walk

![](https://ww1.sinaimg.cn/large/006tNbRwjw1fbj03pbyabj316m0jedjo.jpg)

here :

$w_{vx}$ is the weight for edge $e_{vx}$ 

$d_{tx}$ is the distance between $t$ and $x$

$p$ , control the likelihood of imediately revisiting a node in the walk

$q$ , allows the search to differentiate between 'inward' and 'outward' nodes



