#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[6]:


movies.head()


# In[7]:


credits.head(1)


# In[8]:


movies = movies.merge(credits,on='title')


# In[9]:


movies.head(1)


# In[10]:


# genres
# id 
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[11]:


movies.info()


# In[12]:


movies.head()


# In[13]:


movies.isnull().sum()


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies.duplicated().sum()


# In[16]:


movies.iloc[0].genres


# In[17]:


import ast
def convert(obj):
    L = []

    for i in ast.literal_eval(obj):
        L.append(i['name'])


    return L


# In[18]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#{'Action', 'Adventure', 'FFantasy', 'SciFi'}


# In[19]:


movies['genres'] = movies['genres'].apply(convert)


# In[20]:


movies.head()


# In[21]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[22]:


movies.head()


# In[23]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break

    return L


# In[24]:


import ast

def extract_cast_names(cast):
    if isinstance(cast, str):
        cast = ast.literal_eval(cast)
    elif not isinstance(cast, list):
        return []

    return [actor['name'] for actor in cast[:3]]


# In[25]:


def extract_cast_names(cast):
    cast = ast.literal_eval(cast)
    return [actor['name'] for actor in cast[:3]]


# In[26]:


movies['cast'] = movies['cast'].apply(extract_cast_names)


# In[27]:


movies['cast'].head()


# In[28]:


movies.head()


# In[29]:


movies['crew'][0]


# In[30]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[31]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[32]:


movies.head()


# In[33]:


movies['overview'][0]


# In[34]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[35]:


movies.head()


# In[36]:


movies['genres'] = movies['genres'].apply(lambda x: list(x))
movies['keywords'] = movies['keywords'].apply(lambda x: list(x))
movies['cast'] = movies['cast'].apply(lambda x: list(x))
movies['crew'] = movies['crew'].apply(lambda x: list(x))


# In[37]:


movies.head()


# In[38]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[39]:


movies.head()


# In[40]:


new_df = movies[['movie_id','title','tags']]


# In[41]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[42]:


new_df.head()


# In[45]:


import nltk


# In[44]:


# get_ipython().system('pip install nltk')


# In[46]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[47]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


# In[48]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[49]:


new_df['tags'][0]


# In[50]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[51]:


new_df.head()


# In[52]:


new_df['tags'][0]


# In[53]:


new_df['tags'][1]


# In[54]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[55]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[56]:


vectors


# In[57]:


vectors[0]


# In[58]:


cv.get_feature_names_out()


# In[59]:


['loved','loving','love']
['love','love','love']


# In[60]:


ps.stem('dance')


# In[61]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy Science Fiction culture clash future space war space colony society space travel futuristic romance space alien tribe alien planet cgi marine soldier battle love affair anti war power relations mind and soul 3d Sam Worthington Zoe Saldana Sigourney Weaver James Cameron')


# In[62]:


from sklearn.metrics.pairwise import cosine_similarity


# In[63]:


similarity = cosine_similarity(vectors)


# In[64]:


similarity.shape


# In[65]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[66]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)




# In[67]:


recommend('Batman Begins')


# In[68]:


new_df.iloc[1216]


# In[69]:


new_df.iloc[1216].title


# In[70]:


import pickle
import numpy as np

# Load the original pickle file
with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Save as a memory-mapped NumPy file
np.save('similarity.npy', similarity)
print("Saved similarity.npy successfully!")


# In[71]:


import numpy as np

# Memory-mapped loading (read-only)
similarity = np.load('similarity.npy', mmap_mode='r')


# In[72]:


def get_similar_movies(movie_index, top_n=5):
    sim_scores = similarity[movie_index]  # only loads this row
    top_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return top_indices


# In[ ]:




