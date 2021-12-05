#!/usr/bin/env python
# coding: utf-8

# # LightGCN

# Graphs are versatile data strucutres that can model complex elements and relationships. In this chapter I implement a Light Graph Convolution Network (LightGNC) to make recommendations. This work utilises a recommender library developed by Microsoft, instructions on installation can be found [here](https://github.com/microsoft/recommenders). The library provides utilities to aid common recommendation building tasks such as data cleaning, test/train splitting and the implementation of algorithms.

# ## Outline
#   1. Overview of LightGCN
#   1. Prepare data and hyper-parameters
#   1. Create and train model¶
#   1. Recommendations and evaluation¶

# ## LightGCN Overview & Architecture
# 
# Graph Convolution Network (GCNs) approaches involve semi-supervised learning on graph-structured data. Many real-world datasets come in the form of property graphs, yet until recently little effort has been devoted to the generalization of neural network models to graph structured datasets. GCNs are based on an efficient variant of convolutional neural networks. Convolutional architecure allow the to scale linearly and learn hidden layer representations.
# 
# LightGCN is a simplified design of GCN, more concise and appropriate for recommenders. The model architecture is illustrated below.
# 
# <img src="https://recodatasets.z20.web.core.windows.net/images/lightGCN-model.jpg" width="600">
# 
# 
# In Light Graph Convolution, only the normalized sum of neighbor embeddings is performed towards next layer; other operations like self-connection, feature transformation, and nonlinear activation are all removed, which largely simplifies GCNs. In Layer Combination,the embeddings at each layer are summed over to achieve the final representations.
# 
# ### Light Graph Convolution (LGC)
# 
# In LightGCN, a simple weighted sum aggregator is utilised. The graph convolution operation in LightGCN is defined as:
# 
# $$
# \begin{array}{l}
# \mathbf{e}_{u}^{(k+1)}=\sum_{i \in \mathcal{N}_{u}} \frac{1}{\sqrt{\left|\mathcal{N}_{u}\right|} \sqrt{\left|\mathcal{N}_{i}\right|}} \mathbf{e}_{i}^{(k)} \\
# \mathbf{e}_{i}^{(k+1)}=\sum_{u \in \mathcal{N}_{i}} \frac{1}{\sqrt{\left|\mathcal{N}_{i}\right|} \sqrt{\left|\mathcal{N}_{u}\right|}} \mathbf{e}_{u}^{(k)}
# \end{array}
# $$
# 
# The symmetric normalization term $\frac{1}{\sqrt{\left|\mathcal{N}_{u}\right|} \sqrt{\left|\mathcal{N}_{i}\right|}}$ follows the design of standard GCN, which can avoid the scale of embeddings increasing with graph convolution operations.
# 
# 
# ### Layer Combination and Model Prediction
# 
# The embeddings at the 0-th layer are the only trainable parameters, i.e., $\mathbf{e}_{u}^{(0)}$ for all users and $\mathbf{e}_{i}^{(0)}$ for all items. After $K$ layer, the embeddings are further combined at each layer to arrive at the final representation of a user (an item):
# 
# $$
# \mathbf{e}_{u}=\sum_{k=0}^{K} \alpha_{k} \mathbf{e}_{u}^{(k)} ; \quad \mathbf{e}_{i}=\sum_{k=0}^{K} \alpha_{k} \mathbf{e}_{i}^{(k)}
# $$
# 
# where $\alpha_{k} \geq 0$ denotes the importance of the $k$-th layer embedding in constituting the final embedding. In our experiments, we set $\alpha_{k}$ uniformly as $1 / (K+1)$.
# 
# The model prediction is defined as the inner product of user and item final representations:
# 
# $$
# \hat{y}_{u i}=\mathbf{e}_{u}^{T} \mathbf{e}_{i}
# $$
# 
# which is used as the ranking score for recommendation generation.
# 
# 
# ### Matrix Form
# 
# Let the user-item interaction matrix be $\mathbf{R} \in \mathbb{R}^{M \times N}$ where $M$ and $N$ denote the number of users and items, respectively, and each entry $R_{ui}$ is 1 if $u$ has interacted with item $i$ otherwise 0. The adjacency matrix of the user-item graph is 
# 
# $$
# \mathbf{A}=\left(\begin{array}{cc}
# \mathbf{0} & \mathbf{R} \\
# \mathbf{R}^{T} & \mathbf{0}
# \end{array}\right)
# $$
# 
# Let the 0-th layer embedding matrix be $\mathbf{E}^{(0)} \in \mathbb{R}^{(M+N) \times T}$, where $T$ is the embedding size. Then we can obtain the matrix equivalent form of LGC as:
# 
# $$
# \mathbf{E}^{(k+1)}=\left(\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}\right) \mathbf{E}^{(k)}
# $$
# 
# where $\mathbf{D}$ is a $(M+N) \times(M+N)$ diagonal matrix, in which each entry $D_{ii}$ denotes the number of nonzero entries in the $i$-th row vector of the adjacency matrix $\mathbf{A}$ (also named as degree matrix). Lastly, we get the final embedding matrix used for model prediction as:
# 
# $$
# \begin{aligned}
# \mathbf{E} &=\alpha_{0} \mathbf{E}^{(0)}+\alpha_{1} \mathbf{E}^{(1)}+\alpha_{2} \mathbf{E}^{(2)}+\ldots+\alpha_{K} \mathbf{E}^{(K)} \\
# &=\alpha_{0} \mathbf{E}^{(0)}+\alpha_{1} \tilde{\mathbf{A}} \mathbf{E}^{(0)}+\alpha_{2} \tilde{\mathbf{A}}^{2} \mathbf{E}^{(0)}+\ldots+\alpha_{K} \tilde{\mathbf{A}}^{K} \mathbf{E}^{(0)}
# \end{aligned}
# $$
# 
# where $\tilde{\mathbf{A}}=\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}$ is the symmetrically normalized matrix.
# 
# ### Model Training
# 
# Bayesian Personalized Ranking (BPR) loss is used. BPR is a a pairwise loss that encourages the prediction of an observed entry to be higher than its unobserved counterparts:
# 
# $$
# L_{B P R}=-\sum_{u=1}^{M} \sum_{i \in \mathcal{N}_{u}} \sum_{j \notin \mathcal{N}_{u}} \ln \sigma\left(\hat{y}_{u i}-\hat{y}_{u j}\right)+\lambda\left\|\mathbf{E}^{(0)}\right\|^{2}
# $$
# 
# Where $\lambda$ controls the $L_2$ regularization strength.
# 

# ## Import required packages

# In[176]:


import sys
import os
import papermill as pm
import scrapbook as sb
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams


# ## Read in Data & Set Parameters

# In[173]:


listens = pd.read_csv('.\\data\\processed\\listens.csv',index_col=0)
artists = pd.read_csv('.\\data\\processed\\artists.csv',index_col=0)


# In[167]:


artist_dict = pd.Series(artists.name,index=artists.id).to_dict()


# In[168]:


listens.head(3)


# In[4]:


# top k items to recommend
TOP_K = 10

LISTENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 50
BATCH_SIZE = 1024

SEED = DEFAULT_SEED  # Set None for non-deterministic results

yaml_file = "./lightgcn.yaml"


# ## LightGCN Implementation
# 
# ### Split Data
# We split the full dataset into a train and test dataset to evaluate performance of the algorithm against a held-out set not seen during training. Because SAR generates recommendations based on user preferences, all users that are in the test set must also exist in the training set. We can use the provided python_stratified_split function which holds out a percentage of items from each user, but ensures all users are in both train and test datasets. We will use a 75/25 train/test split. I considered keeping the split at for consistency with the matrix factorization and softmax models. However,this method relies heavily on users' historic listening records and is being split in a different manner so I decided against it. 

# In[174]:


df = listens
df = df.rename(columns={'listenCount': 'rating', 'artistID':'itemID'})
# listens['timestamp'] = np.nan

df.head()


# In[196]:


train, test = python_stratified_split(df, ratio=0.75)


# ### Process data
# 
# `ImplicitCF` is a class that intializes and loads data for the training process. During the initialization of this class, user IDs and item IDs are reindexed, ratings greater than zero are converted into implicit positive interaction, and an adjacency matrix of the user-item graph is created.

# In[197]:


data = ImplicitCF(train=train, test=test, seed=SEED)


# ### Prepare hyper-parameters
# 
# Parameters can be set for ths LightGNC. To save time on tuning parameters we will use the prepared paramemters that can be found in `yaml_file`. `prepare_hparams` reads in the yaml file and prepares a full set of parameters for the model.

# In[198]:


hparams = prepare_hparams(yaml_file,
                          n_layers=3,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          learning_rate=0.005,
                          eval_epoch=5,
                          top_k=TOP_K,
                         )


# ### Create and train model
# 
# With data and parameters prepared, we can create and train the LightGCN model.

# In[199]:


model = LightGCN(hparams, data, seed=SEED)


# In[200]:


with Timer() as train_time:
    model.fit()

print("Took {} seconds for training.".format(train_time.interval))


# ### Recommendations

# `recommend_k_items` produces k artist recommendations for each user passed to the function. `remove_seen=True` removes the artists already listened to by the user. We will produce recommendations using the trained model on instances from the test set as input.

# In[210]:


topk_scores = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)
top_scores = topk_scores
top_scores['name'] = topk_scores.itemID.map(artist_dict)
top_scores.head()


# In[211]:


def user_recommendations(user):
    listened_to = train[train.userID == user].sort_values('rating',ascending=False)
    listened_to['name'] = listened_to.itemID.map(artist_dict)
    listened_to = listened_to.head(10).name
    print('User ' + str(user) + ' most listened to artists...')
    print('\n'.join(listened_to) + '\n')
    
    topk_scores_recs = topk_scores[topk_scores.userID == user].sort_values('prediction',ascending=False).name
    print('User ' + str(user) + ' recommendations...')
    print('\n'.join(topk_scores_recs.tolist()))
    return


# In[212]:


user_recommendations(user=500)


# In[213]:


user_recommendations(user=300)


# At a glance, the recommendation system appears to work extremely well. User 500 has pretty broad and genric music tastes, yet each recommended artist makes sense. User 300 appears to have more specified music interests. Most of user 300's top listened to artists are rock/heavy metal bands from the 70s/80s. The recommendations are also mainly rock/heavy metal bands from the same time period. Across both users, all recommendations appear relevant and potentially useful.

# ### Evaluation
# 
# With `topk_scores` (k=10) predicted by the model, we can evaluate how LightGCN performs on the test set. We will use four evaluation metrics:
# 1. Mean Average Precision (MAP)
# 1. Normalized Discounted Cumulative Gain (NDCGG)
# 1. Precision at 10
# 1. Recall at 10

# In[205]:


eval_map = map_at_k(test, topk_scores, k=TOP_K)
eval_ndcg = ndcg_at_k(test, topk_scores, k=TOP_K)
eval_precision = precision_at_k(test, topk_scores, k=TOP_K)
eval_recall = recall_at_k(test, topk_scores, k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')


# These results are promising and they back up the assumption made from looking at two users' recommendations that the model works. Although, the test split was different than the test splits used to evaluate matrix factorization and softmax, this model's precision is still almost 10 times higher. It appears that this is the superior recommendation system and that we have managed to beat the standard of the initial matrix factorization model.

# ## Conclusion

# LightGCN is a light weight and efficient form of a GCN that can be quickly built, trained, and evaluated on this dataset without the need for a GPU. Even without tuning the hyperparameters, the results and recommendations produced by this model are impressive. Here, we have produced a relevant and potentially useful artist recommendation system. The [recommender library](https://github.com/microsoft/recommenders) was also extremely useful and appropiate for our objective of building an artist recommender system using our Last.fm dataset.
