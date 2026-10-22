🎬 Movie Recommender System
  About the Project

This project is a simple content-based movie recommender system built using Python. It suggests movies that are similar to a selected movie based on their features.

A Streamlit web app is used to make the system interactive and easy to use.

-Features

. Select a movie and get similar recommendations

. Fast recommendations using precomputed similarity matrix

. Simple and interactive Streamlit interface

. Works offline after setup

-Technologies Used

. Python

. Streamlit

. Pandas

. NumPy

. Scikit-learn

. Pickle

📁 Project Files

movie-recommender-system/
│

├── app.py                     # Streamlit web app

├── movies_dict.pkl            # Processed movie dataset

├── similarity.pkl             # Similarity matrix

├── movie-recommender-system.ipynb  # Model development notebook

└── README.md

- How It Works

. Movie data is cleaned and processed

. Important features are combined into text representation

. Vectorization is applied on movie features

. Cosine similarity is calculated between movies

. Based on similarity, top matching movies are recommended

- How to Run the Project
1. Clone the repository
git clone https://github.com/your-username/movie-recommender-system.git
cd movie-recommender-system

2. Install required libraries
pip install streamlit pandas numpy scikit-learn

3. Run the Streamlit app
streamlit run app.py

- Example

If you select:
Movie: Avatar

The system may recommend:

Avatar 2
. John Carter
. Interstellar
. Guardians of the Galaxy
. The Avengers
