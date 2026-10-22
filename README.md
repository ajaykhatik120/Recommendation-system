# 🎬 Movie Recommendation System

A content-based movie recommendation web app built with **Python**, **Pandas**, **NumPy**, **scikit-learn**, and **Streamlit**. Select a movie to discover five similar titles using metadata such as plot overview, genres, keywords, cast, and director.

The project precomputes movie-feature vectors and a cosine-similarity matrix so recommendations can be generated locally without making an API request for every search. Movie poster URLs are cached separately for display.

## Features

- **Content-based recommendations:** recommends movies based on descriptive metadata rather than user ratings or collaborative filtering.
- **Metadata feature engineering:** combines overview text, genres, keywords, top cast members, and director into a single text representation.
- **Text preprocessing:** applies Porter stemming and lowercase normalization.
- **Vectorization:** converts text tags to numerical features with scikit-learn's `CountVectorizer`, limited to 5,000 features and English stop-word removal.
- **Similarity ranking:** uses cosine similarity to retrieve the five highest-scoring movies, excluding the selected title itself.
- **Interactive Streamlit interface:** searchable movie selector, recommendation button, and five poster cards.
- **Cached assets and data:** caches loaded resources with Streamlit and uses pre-fetched poster URLs, avoiding runtime API calls for each recommendation.
- **Memory-mapped similarity matrix support:** loads `similarity.npy` in read-only memory-mapped mode when available, with a `similarity.pkl` fallback.

## How It Works

1. **Load and merge data.** Read the TMDB 5000 Movies and Credits CSV files and merge them on movie title.
2. **Select relevant metadata.** Keep movie ID, title, overview, genres, keywords, cast, and crew; remove rows with missing values.
3. **Extract features.** Parse the structured genre and keyword fields, select up to three cast members, identify the director, and tokenize the overview.
4. **Build tags.** Combine the overview, genres, keywords, cast, and director into one text field per movie.
5. **Normalize text.** Apply Porter stemming and lowercase conversion.
6. **Vectorize.** Use `CountVectorizer(max_features=5000, stop_words="english")` to create text vectors for similarity computation.
7. **Precompute similarity.** Calculate pairwise cosine similarity and store the matrix on disk.
8. **Recommend.** When a user selects a movie, locate its row in the matrix, rank the similarity scores, and display the top five matching titles and cached poster URLs.

### Recommendation logic

For a selected movie, the system compares its feature vector with other movie vectors. Movies with higher cosine similarity scores are ranked higher.

This is a **content-based** recommender: it does not currently learn user preferences, use explicit ratings, or personalize results from user history.

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| Interactive web app | Streamlit |
| Data processing | Pandas, NumPy |
| Text feature extraction | scikit-learn `CountVectorizer` |
| Text preprocessing | NLTK Porter Stemmer |
| Recommendation metric | Cosine similarity |
| Data persistence | Pickle, NumPy `.npy` |
| Movie metadata | TMDB 5000 Movies and Credits datasets |
| Poster metadata | OMDb API (used by the poster-cache build script) |

## Project Structure

```text
Recommendation-system/
├── README.md
└── movie-recommender-system-main/
    ├── app.py                         # Streamlit application
    ├── build_similarity.py            # Preprocesses data and builds similarity artifacts
    ├── build_posters.py               # Fetches and caches poster URLs
    ├── movie-recommender-system.py    # Notebook-exported exploratory implementation
    ├── movie-recommender-system.ipynb # Data exploration and model development
    ├── movies.pkl                     # Processed movie data
    ├── movies_dict.pkl                # Movie IDs and titles used by the app
    ├── posters_cache.pkl              # Cached poster URL mapping
    ├── similarity.pkl                 # Pickled similarity matrix (fallback)
    └── similarity.npy                 # Optional NumPy similarity matrix for memory mapping
```

The CSV datasets used by `build_similarity.py` are expected to be present in the project directory when rebuilding the artifacts. They are not listed in the repository's current project-file listing, so obtain the datasets separately before running preprocessing.

## Prerequisites

- Python 3.9+ recommended
- pip
- The precomputed model artifacts included in the project, or the source CSV datasets if rebuilding them

## Installation

Clone the repository and enter the application directory:

```bash
git clone https://github.com/ajaykhatik120/Recommendation-system.git
cd Recommendation-system/movie-recommender-system-main
```

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the libraries used by the app and preprocessing scripts:

```bash
python -m pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn nltk requests
```

## Run the App

From the `movie-recommender-system-main` directory:

```bash
streamlit run app.py
```

Streamlit prints a local URL (typically `http://localhost:8501`). Open it in your browser, search for or select a movie, and click **Get Recommendations**.

The app expects `movies_dict.pkl` and either `similarity.npy` or `similarity.pkl` in the current working directory. The repository listing includes `movies_dict.pkl` and `posters_cache.pkl`, but does not list either similarity-matrix file. If those files are absent in your clone, run `build_similarity.py` first after supplying the two TMDB CSV datasets described below. The pickle fallback may require substantial memory depending on dataset size.

## Rebuild Recommendation Artifacts

To regenerate the metadata and similarity matrix, place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the application directory and run:

```bash
python build_similarity.py
```

The script creates `similarity.npy`, `similarity.pkl`, `movies_dict.pkl`, and `movies.pkl`.

For the web app's pre-cached poster URLs, run:

```bash
python build_posters.py
```

The poster script queries OMDb by movie title and writes `posters_cache.pkl`. It requires a valid OMDb API key configured in the script. Before using the project beyond local experimentation, move API keys out of source code into environment variables and replace any exposed key.

## Data and Artifacts

- **Input datasets:** TMDB 5000 Movies and Credits CSVs.
- **Processed metadata:** `movies_dict.pkl` contains movie IDs and titles for lookup.
- **Similarity matrix:** `similarity.npy` or `similarity.pkl` stores pairwise movie similarity scores.
- **Poster cache:** `posters_cache.pkl` maps movie IDs to cached poster URLs.

The recommender uses precomputed pairwise similarity, so matrix generation is the most expensive preprocessing stage. At recommendation time, it ranks one similarity row and returns five results.

## Limitations

- **No personalization:** results rely on item metadata, not individual user activity, preferences, or ratings.
- **Pairwise matrix size:** the similarity matrix grows quadratically with the number of movies and can consume significant disk space and memory.
- **Exact title lookup:** selection depends on movie titles in the preprocessed dataset.
- **Feature quality matters:** sparse or incomplete metadata may produce weaker recommendations.
- **Poster availability:** some poster URLs may be missing or invalid; the app uses a placeholder image.
- **Rebuilding requires source datasets:** the TMDB CSV inputs need to be supplied separately if artifacts are regenerated.
- **Security:** do not commit live API keys into source control; use environment variables or a secrets manager.

## Potential Improvements

- Add unit tests for preprocessing, similarity ranking, missing titles, and artifact loading.
- Use sparse vector representations and compute nearest neighbours without storing a full dense similarity matrix.
- Add hybrid recommendations using collaborative signals such as user ratings and interaction history.
- Support metadata filters, movie details, genre selection, and recommendation explanations.
- Add a dependency file with pinned or bounded package versions.
- Move poster API credentials to environment variables and add configurable request retries and rate-limit handling.
- Add dataset/version metadata to generated artifacts so the app can detect mismatched files.

## License

Check the repository for the applicable license before redistributing or reusing the project.
