#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import neo4j
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# In[2]:


listens = pd.read_csv('..\\data\\processed\\listens.csv', index_col=0,encoding='utf-8')
artists_df = pd.read_csv('..\\data\\processed\\artist_info.csv', index_col=0,encoding='utf-8')
artists = pd.read_csv('..\\data\\processed\\artists.csv', index_col=0,encoding='utf-8')
user_friends = pd.read_csv('..\\data\\processed\\user_friends.csv', index_col=0,encoding='utf-8')


# In[3]:


listens.head(3)


# In[4]:


user_friends.head()


# In[5]:


G = nx.Graph()


# In[6]:


G.add_nodes_from(listens.userID.unique())


# In[7]:


G.add_nodes_from(listens.artistID.unique())


# In[8]:


for index,row in user_friends.iterrows():
    G.add_edge(row.userID,row.friendID)


# In[9]:


plt.figure(figsize=(16,16))
nx.draw_networkx(G, with_labels=True, node_size=60, font_size=5)


# In[ ]:




