import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ── Cache ALL data loading — runs only once ever ────────────────────
@st.cache_resource
def load_data():
    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)

    # Memory-mapped numpy loading (near-instant, doesn't load full matrix into RAM)
    try:
        similarity = np.load('similarity.npy', mmap_mode='r')
    except FileNotFoundError:
        similarity = pickle.load(open('similarity.pkl', 'rb'))

    # Load pre-cached poster URLs (no API calls needed at runtime!)
    poster_cache = {}
    if os.path.exists('posters_cache.pkl'):
        poster_cache = pickle.load(open('posters_cache.pkl', 'rb'))

    return movies, similarity, poster_cache

movies, similarity, poster_cache = load_data()

PLACEHOLDER = "https://placehold.co/500x750/1a1a2e/e0e0e0?text=No+Poster"

# ── Recommendation (pure computation, no API calls) ─────────────────
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        # Instant poster lookup from local cache — zero network calls
        poster_url = poster_cache.get(movie_id) or PLACEHOLDER
        recommended_posters.append(poster_url)

    return recommended_movies, recommended_posters

# ── Custom CSS for a polished look ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .movie-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 12px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    .movie-card img {
        border-radius: 12px;
        width: 100%;
    }
    .movie-title {
        color: #e0e0e0;
        font-size: 14px;
        font-weight: 600;
        margin-top: 10px;
        min-height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── UI ──────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Select a movie and discover similar titles instantly</p>', unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    selected_movie = st.selectbox(
        "🔍 Search or select a movie",
        movies['title'].values,
        index=None,
        placeholder="Type a movie name..."
    )
    recommend_btn = st.button('✨ Get Recommendations', use_container_width=True, type="primary")

if recommend_btn and selected_movie:
    names, posters = recommend(selected_movie)

    st.markdown("---")
    st.markdown(f"### Movies similar to **{selected_movie}**")

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.markdown(f'''
            <div class="movie-card">
                <img src="{posters[idx]}" alt="{names[idx]}" onerror="this.src='https://placehold.co/500x750/1a1a2e/e0e0e0?text=No+Poster'">
                <div class="movie-title">{names[idx]}</div>
            </div>
            ''', unsafe_allow_html=True)

elif recommend_btn and not selected_movie:
    st.warning("Please select a movie first!")