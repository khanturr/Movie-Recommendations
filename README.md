# Movie Recommendations

This project implements a movie recommendation system using collaborative filtering techniques with the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

## Project Overview

The goal is to predict user ratings for unseen movies. The dataset contains:
- **100,000 ratings** from **943 users** on **1,682 movies**.
- Ratings range from **1 (worst)** to **5 (best)**.

The project evaluates models using Mean Absolute Error (MAE) and optimizes hyperparameters to improve accuracy and generalizability.

---

## Methods

### 1. Collaborative Filtering with SGD
- Uses stochastic gradient descent (SGD) to optimize matrix factorization.
- Includes regularization to prevent overfitting.

### 2. Singular Value Decomposition (SVD)
- Decomposes the user-movie rating matrix into latent factors.
- Hyperparameters are tuned via grid search.

---

## Results

- **Validation MAE**: `0.7441`
- **Leaderboard MAE**: `0.7450`
- The model performs better for:
  - Users with more ratings.
  - Genres like action, which have more data.

---

## Future Work

- Incorporate additional user and movie features (e.g., age, genre).
- Explore alternative recommendation approaches (e.g., hybrid models, KNN).
