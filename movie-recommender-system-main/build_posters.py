"""
Pre-fetch poster URLs for all movies using OMDb API and cache locally.
This runs ONCE at build time so the app never needs external API calls.
"""
import pickle
import pandas as pd
import requests
import time
import os

OMDB_API_KEY = "b6003d8a"
CACHE_FILE = "posters_cache.pkl"
BATCH_SIZE = 50

# Load movie data
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Load existing cache if available
if os.path.exists(CACHE_FILE):
    poster_cache = pickle.load(open(CACHE_FILE, 'rb'))
    print(f"Loaded existing cache with {len(poster_cache)} entries")
else:
    poster_cache = {}

titles = movies['title'].tolist()
movie_ids = movies['movie_id'].tolist()

total = len(titles)
fetched = 0
skipped = 0
failed = 0

print(f"Total movies: {total}")
print(f"Already cached: {len(poster_cache)}")
print(f"Remaining: {total - len([mid for mid in movie_ids if mid in poster_cache])}")
print("-" * 50)

for idx, (movie_id, title) in enumerate(zip(movie_ids, titles)):
    if movie_id in poster_cache:
        skipped += 1
        continue

    try:
        # Search by title on OMDb
        url = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={OMDB_API_KEY}"
        resp = requests.get(url, timeout=5)

        if resp.status_code == 200:
            data = resp.json()
            poster_url = data.get('Poster', 'N/A')
            if poster_url and poster_url != 'N/A':
                poster_cache[movie_id] = poster_url
                fetched += 1
            else:
                poster_cache[movie_id] = None
                failed += 1
        else:
            poster_cache[movie_id] = None
            failed += 1

    except Exception as e:
        poster_cache[movie_id] = None
        failed += 1

    # Progress update every 50 movies
    if (idx + 1) % BATCH_SIZE == 0:
        print(f"Progress: {idx + 1}/{total} | Fetched: {fetched} | Failed: {failed}")
        # Save intermediate cache
        pickle.dump(poster_cache, open(CACHE_FILE, 'wb'))
        # Small delay to respect rate limits
        time.sleep(0.5)

# Final save
pickle.dump(poster_cache, open(CACHE_FILE, 'wb'))

total_cached = sum(1 for v in poster_cache.values() if v is not None)
print("-" * 50)
print(f"Done! Cached {total_cached}/{total} poster URLs")
print(f"Saved to {CACHE_FILE}")
