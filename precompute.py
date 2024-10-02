#This Code will build a collaborative filter model trained upon genre similarity matrix
#Previously, my results were not as good because it did not consider, the genre and actual film of the movie. 

#Realistically, this wouldn't work because this would take over 55 GBs of RAM, what my PC can't do. So, realistically I could choose a subset of data, but I could also try to move this into a database on SQL
#But in the case below, im using a subset of the data

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the Movie Data
movies = pd.read_csv('movies.csv')

# Select a smaller subset of the data in this case ill be using top 2000
top_n = 2000
movies_subset = movies.head(top_n)

# Create Genre Similarity Matrix for the subset
print(f"Creating genre similarity matrix for top {top_n} movies...")
genre_data = movies_subset['genres'].str.get_dummies('|')
genre_similarity = pd.DataFrame(cosine_similarity(genre_data), index=movies_subset['movieId'], columns=movies_subset['movieId'])

# Save the smaller Genre Similarity Matrix
joblib.dump(genre_similarity, f'genre_similarity_matrix_top_{top_n}.pkl')
print(f"Genre similarity matrix saved as 'genre_similarity_matrix_top_{top_n}.pkl'.")
