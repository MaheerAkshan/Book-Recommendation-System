"""
recommender.py
----------------
This module contains the NLP logic for the Book Recommendation System.

How it works:
1. We load a dataset of books (title, author, genre, description).
2. We combine the text fields (genre + author + description) into one
   "content" string for every book.
3. We use TF-IDF (Term Frequency - Inverse Document Frequency) to convert
   each book's content into a numeric vector.
4. We use Cosine Similarity to measure how similar two books are based on
   their TF-IDF vectors.
5. Given a book title, we return the most similar books.

This is a classic "Content-Based Filtering" recommendation approach,
which relies on Natural Language Processing (NLP) techniques.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "books.csv")


class BookRecommender:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.books_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._load_data()
        self._build_model()

    def _load_data(self):
        """Load the books dataset from CSV into a pandas DataFrame."""
        self.books_df = pd.read_csv(self.data_path)
        self.books_df.fillna("", inplace=True)

        # Combine relevant text columns into a single "content" field.
        # This combined text is what the NLP model will analyze.
        self.books_df["content"] = (
            self.books_df["genre"] + " " +
            self.books_df["author"] + " " +
            self.books_df["description"]
        )

    def _build_model(self):
        """Build the TF-IDF matrix and cosine similarity matrix."""
        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.books_df["content"])

        # cosine_sim[i][j] = similarity score between book i and book j
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_all_titles(self):
        """Return a list of all book titles (used to populate the dropdown)."""
        return self.books_df["title"].tolist()

    def get_recommendations(self, title, top_n=5):
        """
        Given a book title, return the top_n most similar books
        based on cosine similarity of their TF-IDF vectors.
        """
        matches = self.books_df[self.books_df["title"].str.lower() == title.lower()]
        if matches.empty:
            return []

        idx = matches.index[0]

        # List of (index, similarity_score) pairs for the chosen book
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort books by similarity score, highest first
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip the first result (it's the book itself) and take the next top_n
        sim_scores = sim_scores[1:top_n + 1]

        book_indices = [i for i, score in sim_scores]
        results = self.books_df.iloc[book_indices][
            ["title", "author", "genre", "description"]
        ].copy()
        results["similarity"] = [round(score * 100, 1) for i, score in sim_scores]

        return results.to_dict(orient="records")

    def search_books(self, query):
        """
        Simple keyword search across title, author and genre.
        Used for the search bar on the home page.
        """
        query = query.lower()
        mask = (
            self.books_df["title"].str.lower().str.contains(query) |
            self.books_df["author"].str.lower().str.contains(query) |
            self.books_df["genre"].str.lower().str.contains(query)
        )
        return self.books_df[mask][["title", "author", "genre", "description"]].to_dict(
            orient="records"
        )
