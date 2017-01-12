---
layout: post
title: data science winter school note - Network Embedding
date: 2017-01-08
tags: [data science winter school note]
comments: true
share: true
---

> Xin Zhao

> Renmin University of China

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



### GENE

![](https://ww2.sinaimg.cn/large/006tNbRwjw1fbj1snfvz3j316u0oaag2.jpg)

### LINE

- ​First-order Proximity
- Second-order Proximity

### SDNE

- Autoencoder



### Recall Network Embedding Models

- DeepWalk

  - Node sentences + word2vec 

- Node2vec
  - DeepWalk + more sampling strategies 
- GENE
  - Group~document + doc2vec(DM, DBOW) 
- LINE
  - Shallow + first-order + second-order proximity 
- SDNE
  - Deep + First-order + second-order proximity



### Application

- ​Basic Applications
  1. Networkreconstruction
  2. Link prediction
  3. Clustering
  4. Featurecoding
- Data Visualization
- Text Classification
  - free text $\rightarrow$ word co-occurrence network
- Recommendation

---

1. There are no boundaries between data types and research areas in terms of mythologies

2. Even if the ideas are similar, we can move from shallow to deep if the performance actually improves