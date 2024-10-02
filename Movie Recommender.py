# I want to create a movie recommendation system and put it into a Streamlit App. 
# Libraries used: Surprise (to use Prediction Algorithm), scikit learn, import pandas for standard data manipulation and streamlit for to create the App

import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import streamlit as st
import joblib

# Now with Dataset Installed, I will load into load_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    # Choose the top 2000 movies after submodelling it 
    top_n = 2000
    movies = movies.head(top_n)  # Limit movies to top 2000
    ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
    return movies, ratings

# Train Model Based on Ratings, I want to predict what a user who likes x genre will like a similar movie of the same x genre.
@st.cache_data
def load_genre_similarity_matrix():
    genre_similarity = joblib.load('genre_similarity_matrix_top_2000.pkl')
    return genre_similarity

# Recommend movies based on genre similarity
def get_genre_recommendations(movie_id, movies, genre_similarity, n=10):
    # Check if the movie_id is in the filtered movies dataset
    if movie_id not in genre_similarity.index:
        return ["Movie ID not found in the top 2000 movies."]
    
    # Get the similarity scores for the specified movie_id
    similar_scores = genre_similarity[movie_id]
    
    # Sort movies by similarity score, excluding the movie itself
    similar_movies = similar_scores.sort_values(ascending=False).index[1:n + 1]
    
    # Return the recommended movie titles
    recommended_titles = [movies[movies['movieId'] == mid]['title'].values[0] for mid in similar_movies]
    return recommended_titles

# Streamlit UI 
def main():
    st.title("What would you recommend me?")
    st.write("A movie recommendation tool using genre similarity matrix")

    # Load dataset in
    movies, ratings = load_data()
    genre_similarity = load_genre_similarity_matrix()

    st.sidebar.header('User Options')

    # Handle session state for movie ID input
    if 'selected_movie_id' not in st.session_state:
        st.session_state.selected_movie_id = None

    # Lookup movie on the side because how else would the user know?
    st.sidebar.header('Movie Lookup')
    search_type = st.sidebar.radio('Search by:', ('Movie Title', 'Movie ID'))

    if search_type == "Movie Title":
        movie_title = st.sidebar.text_input('Enter Movie Title:')
        if st.sidebar.button('Search by Title'):
            if movie_title:
                search_results = movies[movies['title'].str.contains(movie_title, case=False)]
                st.sidebar.write(f'Found {len(search_results)} movies:')
                st.sidebar.dataframe(search_results[["movieId", "title"]])

    elif search_type == 'Movie ID':
        movie_id_input = st.sidebar.number_input('Enter Movie ID:', min_value=int(movies['movieId'].min()), max_value=int(movies['movieId'].max()))
        if st.sidebar.button('Search by ID'):
            movie_info = movies[movies['movieId'] == movie_id_input]
            if not movie_info.empty:
                st.sidebar.write(f'Movie with ID {movie_id_input}')
                st.sidebar.write(movie_info[['movieId', 'title']])
                st.session_state.selected_movie_id = movie_id_input  # Store the movie ID in session state
        
    # Only show the button if a valid movie ID is selected
    if st.session_state.selected_movie_id is not None:
        if st.sidebar.button('Recommend Similar Movies'):
            st.write(f'Top movie recommendations similar to Movie ID {st.session_state.selected_movie_id}:')
            recommended_movies = get_genre_recommendations(st.session_state.selected_movie_id, movies, genre_similarity)
            for idx, movie in enumerate(recommended_movies, 1):
                st.write(f'{idx}. {movie}')

# Run App
if __name__ == '__main__':
    main()
    
# RUN streamlit run "c:/Users/drake/OneDrive/Desktop/Coding Projects/Youshouldcheckout!/Movie Recommender.py" to look
