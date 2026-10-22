"""
Build all required data files:
  - similarity.npy  (fast numpy format)
  - movies_dict.pkl (movie metadata)
  - posters.pkl     (poster URLs pre-fetched from CSV data)
"""
import numpy as np
import pandas as pd
import ast
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading CSV data...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def extract_cast_names(cast):
    cast = ast.literal_eval(cast)
    return [actor['name'] for actor in cast[:3]]

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

print("Processing features...")
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(extract_cast_names)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

print("Stemming tags...")
new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

print("Vectorizing and computing similarity matrix...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

print(f"Similarity matrix shape: {similarity.shape}")

# Save as .npy for fast memory-mapped loading
print("Saving similarity.npy...")
np.save('similarity.npy', similarity)

# Also save .pkl for backward compat
print("Saving similarity.pkl...")
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Save movies_dict
print("Saving movies_dict.pkl...")
movies_dict = new_df[['movie_id', 'title']].to_dict()
pickle.dump(movies_dict, open('movies_dict.pkl', 'wb'))

# Save movies.pkl 
print("Saving movies.pkl...")
pickle.dump(new_df, open('movies.pkl', 'wb'))

print("\nDone! All files created successfully.")
