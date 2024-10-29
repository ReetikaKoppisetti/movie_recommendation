# app.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify, render_template


app = Flask(__name__, static_url_path='', static_folder='')

# Load the dataset from CSV file
df = pd.read_csv('movies.csv')

# Preprocessing: Combine title and genres into one string
df['title_genres'] = df['title'] + ' ' + df['genres']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title_genres'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    genres_input = data.get('genres', '')  # Get 'genres' from JSON data, default to empty string
    
    # Check if genres_input is empty or contains invalid characters
    if not genres_input or not all(word.isalpha() or word == ',' for word in genres_input.split(',')):
        return jsonify({"error": "Please enter correct genre values."}), 400
    
    # Rest of the code for recommendation generation remains the same...

    
    def recommend_movies(genres, df, tfidf_matrix, n=5):
        # Split the genres string into individual genres
        genre_list = genres.split(',')

        # Combine the new genres with the title for the new movie
        new_movie = "New Movie " + ','.join(genre_list)

        # Transform the new movie's genres using the trained TF-IDF vectorizer
        new_movie_tfidf = tfidf_vectorizer.transform([new_movie])

        # Calculate cosine similarity between the new movie and all movies in the dataset
        cosine_similarities = linear_kernel(new_movie_tfidf, tfidf_matrix).flatten()

        # Get indices of movies sorted by similarity score
        related_movie_indices = cosine_similarities.argsort()[::-1]

        # Get top n recommended movies
        top_movies_indices = related_movie_indices[1:n+1]  # Exclude the new movie itself
        recommended_movies = df.iloc[top_movies_indices]['title'].tolist()

        return recommended_movies
    
    recommended_movies = recommend_movies(genres_input, df, tfidf_matrix)
    return jsonify(recommended_movies)



if __name__ == '__main__':
    app.run(debug=True)
