---
layout: post
title: Kaggle Link Prediction 
date: 2017-01-11
tags: [Data Competition] 
comments: true
share: true
---


#### Competition Detail

> AXA Data Science Winter School : Tsinghua, Renmin and Ecole Polytechnique
>
> Link prediction is a task with multidisciplinary applications among which are bioinformatics, social networks and online stores.
>
> On this data challenge, you are given a citation network as a graph where nodes are research papers and there is an edge between two nodes if one of the two papers cite the other. From this citation graph edges have been removed at random.
>
> Your task is to reconstruct the full network as a form of link prediction  using graph-theoretical, textual, and other information. You are being provided with starting code that displays this task as a classification problem of classifying  whether to nodes share a link or not (0,1 class assignments).
>
> You may build the partial graph from the training data and utilize paper information (title, abstract, publication year) to extract features from both.

#### Data

> **training_set.txt**  - 615,512 labeled node pairs (1 if there is an edge between the two nodes, 0 else). One pair and label per row, as: source node ID, target node ID, and 1 or 0. The IDs match the papers in the node_information.csv file (see below).
>
> **testing_set.txt** - 32,648 node pairs. One pair per row, as: source node ID, target node ID.
>
> **node_information.csv** - for each paper out of 27,770, contains the paper (1) unique ID, (2) publication year (between 1993 and 2003), (3) title, (4) authors, (5) name of journal (not available for all papers), and (6) abstract. Abstracts are already in lowercase, common English stopwords have been removed, and punctuation marks have been removed except for intra-word dashes.

---

#### Solution

For this problem we extract **three kinds of features**

1. Text Feature
   - LSI Vector (used for calculating similarity)
   - TFIDF Vector (used for calculating similarity)
   - Common Words
2. Network Feature
   - Degree
   - Between Centrality 
   - PageRank
   - Common  Neighbors
   - Paper Community Category
3. Author and Journal Feature
   - Common Author
   - Delta Year
   - author_paper_year_mean
   - author_paper_num_year
   - journal_paper_num_year
   - journal_class

After try several classification algorithm , we choose **Random Forest** to do prediction.

---

#### Code

Below are main code for feature extraction 

more detail can be find in my  github  : [Link Prediction](https://github.com/ZJCODE/Data_Competition/tree/master/Link%20Prediction)

##### Text Feature

```python
stpwds = set(nltk.corpus.stopwords.words("english"))

node_information = pd.read_csv('./Data/node_information.csv',names=['id','year','title','author','journal','abstract'])

titles = list(node_information.title.values)
ids = list(node_information.id.values)
abstracts = list(node_information.abstract.values)

title_num_topics = 150
abstract_num_topics = 150

# Title
titles = [[w for w in t.lower().split() if (w not in stpwds) ] for t in titles]
dictionary_title = corpora.Dictionary(titles)
corpus_title = [dictionary_title.doc2bow(text) for text in titles]
tfidf_title = models.TfidfModel(corpus_title)
corpus_tfidf_title = tfidf_title[corpus_title]
lsi = models.LsiModel(corpus_tfidf_title, id2word=dictionary_title, num_topics=title_num_topics) # initialize an LSI transformation
corpus_lsi_title = lsi[corpus_tfidf_title] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

title_feature = ['title_feature_'+str(i) for i in range(title_num_topics)]
title_vector = pd.DataFrame([[i[1] for i in a] for a in list(corpus_lsi_title)],columns = title_feature)

# dict
id_title_vector_dict = dict(zip(ids,[[i[1] for i in a] for a in list(corpus_lsi_title)]))

# Abstract
abstracts = [[w for w in t.lower().split() if (w not in stpwds) ] for t in abstracts]
dictionary_abstract = corpora.Dictionary(abstracts)
corpus_abstract = [dictionary_abstract.doc2bow(text) for text in abstracts]
tfidf_abstract = models.TfidfModel(corpus_abstract)
corpus_tfidf_abstract = tfidf_abstract[corpus_abstract]
lsi = models.LsiModel(corpus_tfidf_abstract, id2word=dictionary_abstract, num_topics=abstract_num_topics) # initialize an LSI transformation
corpus_lsi_abstract = lsi[corpus_tfidf_abstract] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

abstract_feature = ['title_fea111ture_'+str(i) for i in range(abstract_num_topics)]
abstract_vector = pd.DataFrame([[i[1] for i in a] for a in list(corpus_lsi_abstract)],columns = abstract_feature)

# dict
id_abstract_vector_dict = dict(zip(ids,list(corpus_lsi_abstract)))



# Common Words [define in get_all_feature function]

for r in relation:
    source,target = r
    try:
        source_title = [w for w in id_title_dict[source].lower().split() if w not in stpwds]
        target_title = [w for w in id_title_dict[target].lower().split() if w not in stpwds]
        comm_word_title.append(len(set(source_title)&set(target_title))*1.0/len(set(source_title)|set(target_title)))
    except:
        comm_word_title.append(0)
        
    
    try:
        source_abstract = [w for w in id_abstract_dict[source].lower().split() if w not in stpwds]
        target_abstract = [w for w in id_abstract_dict[target].lower().split() if w not in stpwds]
        comm_word_abstract.append(len(set(source_abstract)&set(target_abstract))*1.0/len(set(source_abstract)|set(target_abstract)))
    except:
        comm_word_abstract.append(0)
```

##### Network Fetaure

```python
train = pd.read_table('./Data/training_set.txt',sep=' ',names=['source','target','link'])

G = nx.Graph()
train_link = train[train.link == 1][['source','target']]
G.add_edges_from(train_link.values)

id_degree_dict = G.degree()
id_bc_dict = nx.betweenness_centrality(G)
id_cluster_dict = nx.clustering(G)
id_pagerank_dict = nx.pagerank(G)

#------------------------------------

# Community for Paper

def Sort_Dict(Diction):
    L = list(Diction.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L

def DrawGraph(G):
    plt.rc('figure' ,figsize = (15,15))
    nx.draw_networkx(G, pos=nx.spring_layout(G), arrows=True, with_labels=False, node_size=1,node_color='r')

def GetCoreSubNetwork(G,start = False,end = False):    
    G_UnDi = G.to_undirected()
    D = nx.degree(G_UnDi)
    SD = Sort_Dict(D)
    if end == False and start == False:
        Sample_Nodes = [a[0] for a in SD[:]]
    else:
        Sample_Nodes = [a[0] for a in SD[start:end]]
    SubG = nx.subgraph(G_UnDi,Sample_Nodes)
    return SubG
    
def CommunityDetection(DG,n,draw = False ,with_arrow = False,with_label = False):
    
    G = DG.to_undirected()
    Community_Nodes_List = []
    if draw == True:        
        plt.rc('figure',figsize=(12,10))
    #first compute the best partition
    partition = community.best_partition(G) # Nodes With Community tag
    from collections import Counter
    Main = [a[0] for a in Sort_Dict(Counter(partition.values()))[:n]] # Top n Community's Tag
    ZipPartition = partition.items()
    SubNodes = [a[0] for a in ZipPartition if a[1] in Main] # NodesList belong to Top Community
    
    if with_arrow == False:        
        SubG = nx.subgraph(G,SubNodes)
    else:
        SubG = nx.subgraph(DG,SubNodes)
        
    #pos = nx.spectral_layout(SubG)
    #pos = nx.spring_layout(SubG)
    #pos = nx.shell_layout(SubG)
    pos = nx.fruchterman_reingold_layout(SubG)
    if draw == True:        
        if with_label == True:        
            nx.draw(SubG,pos,node_size = 1,alpha =0.1,with_labels=True)
    #drawing
    count = -1

    color = ['b','g','r','c','m','y','k','w']
    for com in set(partition.values()) :
        if com in Main:
            count = count + 1
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            Community_Nodes_List += zip(list_nodes,[count]*len(list_nodes))
            if draw == True:
                nx.draw_networkx_nodes(SubG,pos, list_nodes, node_size = 60,
                                        node_color = color[count],alpha =0.4,with_labels=True)
    if draw == True:  
        if with_label == True:
            plt.legend(['','']+range(1,n+1))
        else:        
            plt.legend(range(1,n+1))      
        nx.draw_networkx_edges(SubG,pos,arrows=True,alpha=0.2)
        plt.show()
        
    Nodes_Category = pd.DataFrame(Community_Nodes_List,columns=['Id','category'])
    Edges = SubG.edges()
    return Nodes_Category , Edges

SubG = GetCoreSubNetwork(G)

N,E = CommunityDetection(SubG,5)

def Save_DataFrame_csv(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File,encoding='utf8',header=True,index = False)

Save_DataFrame_csv(N,'./Data/papers_network_community_category')

papers_network_community_category_dict = dict(N.values)

# Common Neighbors [define in get_all_feature function]

for r in relation:
    source,target = r
    try:
        comm_neighbor.append(len(set(G.neighbors(source))&set(G.neighbors(target))))
    except:
        comm_neighbor.append(0)

```

##### Author and Journal Feature

```python
node_information = pd.read_csv('./Data/node_information.csv',names=['id','year','title','author','journal','abstract'])

top_group_journal = pd.read_csv('./Data/top_group_journal.txt',names=['journal']).journal.values

author_mean_year = node_information.pivot_table(values='year',index='author',aggfunc='mean')
author_paper_num_year = node_information.pivot_table(values='id',index=['author','year'],aggfunc='count')
journal_paper_num_year = node_information.pivot_table(values='id',index=['journal','year'],aggfunc='count')


author_paper_year_mean_list = []
author_paper_num_year_list = []
journal_paper_num_year_list = []
journal_class = []
id_list = []


for i in range(len(node_information)):
	print 'process :' + str(i) + 'th line'
	info = node_information.ix[i,:]
	
	id_list.append(info.id)

	if info.journal in top_group_journal:
		journal_class.append(1)
	else:
		journal_class.append(0)
	try:                        
		author_paper_year_mean_list.append(author_mean_year[info.author])
	except:                           
		author_paper_year_mean_list.append(np.nan)
	try:
		author_paper_num_year_list.append(author_paper_num_year[info.author][info.year])
	except:
		author_paper_num_year_list.append(np.nan)
	try:
		journal_paper_num_year_list.append(journal_paper_num_year[info.journal][info.year])
	except:
		journal_paper_num_year_list.append(np.nan)


author_journal_feature = pd.DataFrame({'id':id_list,
	'author_paper_year_mean':author_paper_year_mean_list,
	'author_paper_num_year':author_paper_num_year_list,
	'journal_paper_num_year':journal_paper_num_year_list,
	'journal_class':journal_class})
```

##### Predict

```python
# define a function called get_all_feature for combine those features above

feature_test = get_all_feature(relation_test,G)
feature_train = get_all_feature(relation_train,G)
label_train = train.link.values

rfc = RandomForestClassifier(n_estimators=200, random_state=0)
rfc.fit(feature_train,np.array(label_train))  
pred = rfc.predict(feature_test)
print 'write submission file '
submission = pd.DataFrame({'Id':range(len(pred)),'prediction':pred})
submission.to_csv('submission.csv',header=True,index= False)

```

##### Result

```
Can Reach F1 Score Around 0.97520
```



