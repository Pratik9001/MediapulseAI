🎬 MediapulseAI

An AI-powered movie analytics and recommendation platform. Explore dataset statistics, get personalized movie recommendations via Machine Learning, and analyze audience sentiment using NLP — all from a single interactive web app.


Table of Contents

Overview

Features

Project Structure

Dataset Format

Prerequisites

Installation

Running the App

TMDB API Key Setup

App Tabs & Usage

How the ML Models Work

Troubleshooting

Dependencies


Overview
MediapulseAI is a application that ingests a movie dataset (movies_data.csv) and provides:

An interactive analytics dashboard with charts and clickable visualizations
A dual-mode recommender engine (Content-Based Filtering & Collaborative Filtering)
An NLP sentiment analysis engine powered by TextBlob
A searchable raw data explorer with download support


Features
FeatureDescription📊 DashboardMovie ratings distribution (clickable bars), release timeline, top cast actors🤖 Content-Based RecommenderTF-IDF + Cosine Similarity on synopsis, genre, and director metadata👥 Collaborative FilteringUser-based filtering using shared review history🎭 Sentiment AnalysisPer-movie NLP polarity scoring and sentiment distribution charts🔍 Global SearchFilter by title, genre, director, or cast member with poster grid display🎞️ TMDB Poster FallbackAuto-fetches movie posters from The Movie Database (TMDB) API when the CSV poster URL is missing🗄️ Raw Data ExplorerFull dataset view with searchable, scrollable dataframe

Project Structure
mediapulse-ai/
│
├── un_v1.py            # Main Streamlit application (entry point)
├── movies_data.csv     # Movie dataset (required — must be in the same directory)
├── requirements.txt    # Python dependencies
└── README.md           # This file

⚠️ Important: movies_data.csv must be placed in the same directory as un_v1.py. The app uses a relative path (pd.read_csv('movies_data.csv')) to load data.


Dataset Format
The app expects movies_data.csv to contain the following columns:
ColumnTypeDescriptionExampletitlestrMovie title"Inception"yearintRelease year2010ratingstrRating string with numeric value"4.5/5"genresstr (list literal)Stringified Python list of genres"['Action', 'Sci-Fi']"directorsstr (list literal)Stringified Python list of director names"['Christopher Nolan']"caststr (list literal)Stringified Python list of cast members"['Leonardo DiCaprio', 'Joseph Gordon-Levitt']"synopsisstrMovie description / plot summary"A thief who steals corporate secrets..."poster_urlstrURL to the movie poster image"https://image.tmdb.org/..."reviewsstr (list of dicts literal)Stringified Python list of review dicts"[{'username': 'user1', 'review_text': 'Great!', 'likes': 10}]"
Review dict structure (inside the reviews column):
python{
    "username": "string",       # Reviewer's username
    "review_text": "string",    # Written review content
    "likes": int                # Number of likes on the review
}

The app uses ast.literal_eval() to safely parse the stringified lists and dicts. Ensure these columns are formatted as valid Python literals.


Prerequisites

Python 3.9 or higher (Python 3.10+ recommended)
pip (Python package installer)
An active internet connection (for TMDB API poster fetching)

Check your Python version:
bashpython --version
# or
python3 --version

Installation
Step 1 — Clone or download the project
Place all files (un_v1.py, movies_data.csv, requirements.txt) in a single project folder:
bashmkdir mediapulse-ai
cd mediapulse-ai
# Copy your files here
Step 2 — Create a virtual environment (recommended)
Using a virtual environment prevents dependency conflicts with other Python projects on your machine.
bash# Create the virtual environment
python -m venv venv

# Activate it — macOS / Linux
source venv/bin/activate

# Activate it — Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate it — Windows (PowerShell)
venv\Scripts\Activate.ps1
Step 3 — Install dependencies
bashpip install -r requirements.txt
The requirements.txt installs:
streamlit
pandas
numpy
scikit-learn
textblob
plotly

ℹ️ The app also uses requests and ast — both are part of Python's standard library or installed alongside other packages. No extra install is needed.

Step 4 — Download TextBlob corpora
TextBlob requires additional language corpora for NLP. Run this once after installing the packages:
bashpython -m textblob.download_corpora
If that command fails, try:
bashpython -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

Running the App
From inside your project directory (with your virtual environment activated):
bash streamlit run un_v1.py
Streamlit will start a local server and print output like:
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
Open http://localhost:8501 in your browser. The app will load automatically.
To stop the server, press Ctrl + C in your terminal.

TMDB API Key Setup
The app uses the TMDB (The Movie Database) API to fetch movie poster images when a valid poster URL is not found in the CSV. The key is currently hardcoded in the script:
pythonTMDB_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
Using your own TMDB API key (recommended for production)

Create a free account at https://www.themoviedb.org/
Go to Settings → API and request an API key (free for non-commercial use)
Create a secrets file for Streamlit:

bashmkdir -p .streamlit
touch .streamlit/secrets.toml

Add your key to .streamlit/secrets.toml:

tomltmdb_api_key = "YOUR_ACTUAL_API_KEY_HERE"

In un_v1.py, replace the hardcoded key line with:

pythonTMDB_API_KEY = st.secrets["tmdb_api_key"]

If no TMDB API key is configured and a poster URL in the CSV is invalid, the app will display a placeholder image automatically — the app will not crash.


App Tabs & Usage
📊 Tab 1 — Dashboard
Displays high-level statistics about the loaded dataset:

KPI Metrics: Total movies, total reviews extracted, unique reviewers
Ratings Distribution (Interactive): Click any bar in the chart to filter and display all movies with that rating. Alternatively, use the dropdown selector.
Movies Released Over the Years: A line chart showing release volume per year
Most Frequently Cast Actors: A bar chart of the top 10 most-cast actors across all movies

🤖 Tab 2 — Recommender Engine
Contains two sub-tabs:
Global Search:
Filter the movie catalog by any combination of:

Movie title (partial text match)
Genre (multi-select)
Director (single-select dropdown)
Cast member (partial text match)

Results are shown as a 4-column poster grid. Click the ℹ️ button below any poster to view full movie details.
AI Recommender:
Choose between two recommendation modes using the radio button:

Content-Based Filtering: Select a movie you liked. The engine uses TF-IDF vectorization on synopsis + genre + director text to compute cosine similarity and returns the 5 most similar movies.
Collaborative Filtering: Select a user profile. The engine finds users who reviewed similar movies and recommends movies those users also liked but the selected user hasn't seen yet.

🎭 Tab 3 — Sentiment Analysis
Select a movie from the dropdown (only movies with non-empty text reviews appear). Click "Analyze Audience Mood" to:

Run TextBlob polarity scoring on all reviews for that movie
View Overall Audience Mood (Positive 😊 / Neutral 😐 / Negative 😠)
See the Average Polarity Score (range: -1.0 to +1.0)
Explore a Sentiment Distribution Pie Chart and a Polarity Score Histogram
Browse a color-coded review table showing each reviewer's sentiment and score

🗄️ Tab 4 — Raw Data
Displays the cleaned dataset in a scrollable, full-height table. Columns shown: title, year, rating, genres, directors, synopsis.

How the ML Models Work
Content-Based Filtering (TF-IDF + Cosine Similarity)

A combined_features column is built by concatenating each movie's synopsis, genres, and directors into a single text string.
Scikit-learn's TfidfVectorizer converts this corpus into a TF-IDF matrix, weighting terms by importance across the dataset.
Cosine similarity is computed between all movie pairs, producing an N×N similarity matrix.
When a user selects a movie, the app retrieves the row for that movie from the matrix and returns the top-N most similar entries.

Collaborative Filtering (User-Based)

All reviews are flattened into a reviews_df table with (username, movie) pairs.
For a selected user, the app finds all movies they've already reviewed.
It identifies other users who reviewed those same movies (treating them as "similar users").
It collects movies reviewed by similar users that the target user has not seen.
Movies are ranked by review frequency among similar users and the top-N are returned.

Sentiment Analysis (TextBlob)

For each text review, TextBlob(text).sentiment.polarity returns a float from -1.0 (most negative) to +1.0 (most positive).
Reviews are bucketed: > 0.1 → Positive, < -0.1 → Negative, otherwise Neutral.
Aggregated metrics and per-review scores are displayed in the UI.


Troubleshooting
FileNotFoundError: movies_data.csv
The CSV file must be in the same directory as un_v1.py. Run streamlit run un_v1.py from that directory.
ModuleNotFoundError: No module named 'textblob'
Your virtual environment may not be activated, or installation failed. Run:
bashpip install -r requirements.txt
python -m textblob.download_corpora
LookupError: Resource punkt not found
TextBlob corpora are missing. Fix with:
bashpython -m textblob.download_corpora
Posters not loading / showing placeholder
Check your internet connection. If the TMDB API key is invalid or expired, replace it with your own (see TMDB API Key Setup).
ast.literal_eval errors on startup
A value in the genres, directors, cast, or reviews columns is not a valid Python list literal. Inspect the offending row in the CSV and ensure list columns use proper Python syntax: "['item1', 'item2']".
Streamlit version conflicts (on_select not recognized)
The Dashboard's clickable chart requires Streamlit 1.35+. Upgrade with:
bashpip install --upgrade streamlit

Dependencies
PackagePurposestreamlitWeb UI framework and interactive widget renderingpandasData loading, transformation, and filteringnumpyNumerical computations (polarity averaging, etc.)scikit-learnTF-IDF vectorization and cosine similaritytextblobNLP sentiment polarity scoringplotlyInteractive charts (bar, pie, line, histogram)requestsHTTP calls to the TMDB API for poster imagesastSafe parsing of stringified Python lists/dicts in CSV

Built with ❤️ using Streamlit, scikit-learn, and TextBlob.
