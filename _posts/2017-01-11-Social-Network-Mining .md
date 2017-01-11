---
layout: post
title: data science winter school note - Social Network Mining 
date: 2017-01-11
tags: [data science winter school note]
comments: true
share: true
---

> Jie Tang

> Tsinghua University

Social Network Mining  — Computational models for mining big social networks

### What is Social Network ?

Trace Back : 

1. Web : 
   - Hyperlinks between web pages [PageRank / Hints]
2. Collaborative Web.  (Info. Space + Users)
   - Personalized recommendation
   - Collaborative Filtering
3. Social Network. (Info. Space + Social Space)

### History

- Six Degree of Separation [1967]
- Weak Tie [1973]
- Dunbar's Number [1992]
- Structural Hole [1995]
- Hubs & Authorities [1997]
- Small World [1998]
- Scale Free [1999]
- Community Detection [2002]
- Link Prediction , Influence Maximization
- Densification[2005]
- Spread of Obesity, Smoking, Happiness[2007-2009]
- Computational Social Science [2009-2012]

> A field is emerging that leverages the capacity to collect and analyze data at a scale that may reveal patterns of individual and group behaviors



### Network Roadmap

- Socail Theories
- Graph. Theories

|    Part 1     |      Part 2       |     part 3      |
| :-----------: | :---------------: | :-------------: |
| User Modeling | Socail Tie 、 Line | Triad Formation |
|  Demographic  |     Homophily     |    Community    |
|  Social Role  | Social Influence  | Group Behavior  |

> User $\rightarrow$ Tie $\rightarrow$ Structure

### How to model a user in social network

-  Profiling
-  Preference Mining
-  Demographics
-  More ?

##### Traditional way :

1. Source Finding
2. Extraction

##### New Way Basic Idea ：

1. A uniform framework
   - All in one step, avoiding error propagation
   - Incorporate information from different data sources: Homepage, Google Scholar, Twitter, Linkedin, Facebook, etc.

2. Use search engine as the data source

3. Smart Query Construction




##### Markov Logic Factor Graph

Why logic factors?

1. Depict and utilize correlations between possible candidates from redundant data.
2. Incorporate human knowledge to guide and amend the classification model.



- Complete Consistency
- Partial Consistency

- Prior Knowledge



##### Connecting Multiple Networks

Basic Idea:

Identifying users from multiple heterogeneous networks and integrating semantics from the different networks together.

Local vs. Global consistency

- Local matching : matchingusersbyprofiles
- Network matching : matchingusers’egonetworks
- Global consistency : matchingusersbyavoiding global inconsistency

### Socail Tie Analysis

##### 1. Inferring Social ties

- User $\rightarrow$ Node

- Relation $\rightarrow$ Node



> Inferring Social Ties Across Networks

Social Theories

- Social balance theory
  - The underlying networks are unbalanced
  - While the friendship networks are balanced
- Structural hole theory
  - Users are more likely (+25-150% higher than change) to have the same type of relationship with C if C spans structural holes
- Social status theory
  - 99% of triads in the networks satisfy the social status theory
- Two-step-flow theory
  - Opinion leaders are more likely (+71%-84% higher than chance) to have a higher social-status than ordinary users.


##### 2. Reciprocity

- Consider Geographic Distance
- Link homophily: users who share common links will have a tendency to follow each other.
- Status homophily: Elite users have a much stronger tendency to follow each other.
- Interaction
- Structural balance
  - Reciprocalrelationshipsare balanced (88%)
  - Parasocialrelationshipsare not (only 29%)

##### 3. Triadic Closure

### How people influence each other

- From text sentiment to user sentiment
- From user sentiment to network sentiment

Learning for network sentiment analysis

- Semi-supervised Factor Graph Model

Topic-based Social Influence Analysis

- The Solution: Topical Affinity Propagation
- Topical Factor Graph (TFG) Model

### Real Applications 

Big Data Analytics in Game Data

- DNF & QQ Speed

### Information Diffusion

Structural hole spanners control the information diffusion

Structural hole spanners are more likely to connect important nodes in different communities.

- If a user is connected with many opinion leaders in different communities, more likely to span structural holes.
- If a user is connected with structural hole spanners , more likely to act as an opinion leader.


##### Model 

1. HIS
2. MaxD






