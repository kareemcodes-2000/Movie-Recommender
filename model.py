import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import joblib

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Train the model
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
algo = SVD()
algo.fit(trainset)

# Save the model using joblib
joblib.dump(algo, 'svd_model_full.pkl')
print("Model saved as 'svd_model_full.pkl'")