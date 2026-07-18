"""
app.py
------
Flask web application for the NLP-based Book Recommendation System.

Routes:
  /                -> Home page: shows all books + search bar + dropdown to pick a book
  /search  (POST)  -> Handles keyword search
  /recommend (POST)-> Handles "Recommend similar books" for a chosen title
"""

from flask import Flask, render_template, request
from recommender import BookRecommender

app = Flask(__name__)

# Build the recommendation model once when the server starts.
engine = BookRecommender()


@app.route("/")
def home():
    titles = engine.get_all_titles()
    all_books = engine.books_df[["title", "author", "genre", "description"]].to_dict(
        orient="records"
    )
    return render_template("index.html", titles=titles, books=all_books)


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    titles = engine.get_all_titles()

    if not query:
        return render_template("index.html", titles=titles, books=[], query=query)

    results = engine.search_books(query)
    return render_template("index.html", titles=titles, books=results, query=query)


@app.route("/recommend", methods=["POST"])
def recommend():
    selected_title = request.form.get("book_title")
    titles = engine.get_all_titles()

    recommendations = engine.get_recommendations(selected_title, top_n=5)

    return render_template(
        "recommendations.html",
        selected_title=selected_title,
        recommendations=recommendations,
        titles=titles,
    )


if __name__ == "__main__":
    # debug=True gives auto-reload + error pages, handy for a college project demo
    app.run(debug=True)
