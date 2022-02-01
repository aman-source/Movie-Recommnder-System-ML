#!/usr/bin/env python
# coding: utf-8

# In[2]:


# tmdb_5000_movies.csv


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("tmdb_5000_movies.csv")


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.columns


# In[22]:


features = ['keywords','production_companies','genres']
for feature in features:
    df[feature] = df[feature].fillna('')


# In[23]:


def combine_feature(row):
    try:
        
        return row['keywords'] + " "+ row['production_companies'] + " "+ row['genres'] 
    except:
        print(row)

df['combined_features']=df.apply(combine_feature, axis =1)


# In[24]:


df.iloc[0]['combined_features']


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[27]:


cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
cosine_model = cosine_similarity(count_matrix)


# In[28]:


cosine_model_df = pd.DataFrame(cosine_model)
cosine_model_df.head()


# In[29]:


cosine_model_df = pd.DataFrame(cosine_model, index = df.title, columns = df.title)
cosine_model_df.head()


# In[32]:


def make_recommendations(movie_user_likes):
    return cosine_model_df[movie_user_likes].sort_values(ascending=False)[:20]

make_recommendations('Bang')


# In[33]:


make_recommendations('Spider-Man 3')


# In[34]:


make_recommendations('Men in Black 3')


# In[35]:


make_recommendations('Titanic')


# In[ ]:




