import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Step 1: Load datasets
ratings_file = 'data_movie_lens_100k/ratings_all_development_set.csv'
leaderboard_file = 'data_movie_lens_100k/ratings_masked_leaderboard_set.csv'
movie_info_file = 'data_movie_lens_100k/movie_info.csv'
user_info_file = 'data_movie_lens_100k/user_info.csv'

ratings_data = pd.read_csv(ratings_file)
leaderboard_data = pd.read_csv(leaderboard_file)
movie_info = pd.read_csv(movie_info_file)
user_info = pd.read_csv(user_info_file)

# Display the first few rows of the datasets
print("Ratings Data:\n", ratings_data.head())
print("Leaderboard Data:\n", leaderboard_data.head())
print("Movie Info:\n", movie_info.head())
print("User Info:\n", user_info.head())

# Step 2: Prepare ratings data for the surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['user_id', 'item_id', 'rating']], reader)

# Step 3: Split data into training and validation sets
trainset, validset = train_test_split(data, test_size=0.2)

# Step 4: Define the chosen parameters for the SVD model
n_factors = 50  # Latent factors (chosen based on typical performance)
n_epochs = 20  # Number of epochs (commonly used to balance training time and performance)
lr_all = 0.005  # Learning rate (moderate value for convergence)
reg_all = 0.1  # Regularization term (helps avoid overfitting)

# Step 5: Initialize and train the SVD model with the chosen parameters
model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
model.fit(trainset)

# Step 6: Evaluate the model on the validation set
predictions = model.test(validset)
mae = accuracy.mae(predictions, verbose=True)
print("Validation MAE:", mae)

# Step 7: Make predictions for the leaderboard set
# Generate predictions for leaderboard user-item pairs
predictions = leaderboard_data.apply(
    lambda row: model.predict(row['user_id'], row['item_id']).est, axis=1
)

# Step 8: Save predictions to a plain text file
# The predictions are saved as a 1D array of floats, one per line
with open('predicted_ratings_leaderboard.txt', 'w') as f:
    for pred in predictions:
        f.write(f"{pred}\n")

print("Predictions saved to 'predicted_ratings_leaderboard.txt'.")