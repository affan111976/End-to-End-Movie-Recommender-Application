import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 4 

def create_sequences():
    print("Loading ratings data with timestamps...")
    ratings_df = pd.read_csv('data/ratings.csv', parse_dates=['created_at'])
    # print(ratings_df)

    movie_ids = ratings_df['movie_id'].unique().tolist()
    # print(movie_ids)
    movie_to_idx = {movie: i + 1 for i, movie in enumerate(movie_ids)} 
    print(movie_to_idx)
    idx_to_movie = {i + 1: movie for i, movie in enumerate(movie_ids)}
    print(idx_to_movie)
    n_unique_movies = len(movie_ids) + 1 

    print(f"Found {n_unique_movies - 1} unique movies.")

    # Sort interactions chronologically for each user
    ratings_df.sort_values(by=['user_id', 'created_at'], inplace=True)
    
    # Map movie IDs to their new indices
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie_to_idx)
    # print(ratings_df)
    
    # Group movies by user to form their history sequences
    user_histories = ratings_df.groupby('user_id')['movie_idx'].apply(list)
    print(user_histories)
    print(type(user_histories))
    
    
    sequences = []
    targets = []
    
    print("Generating sequences using a sliding window")
    for history in user_histories:
        if len(history) > SEQUENCE_LENGTH:
            for i in range(len(history) - SEQUENCE_LENGTH):
                seq = history[i:i + SEQUENCE_LENGTH]
                print(seq)
                target = history[i + SEQUENCE_LENGTH]
                print(target)
                sequences.append(seq)
                targets.append(target)

    print(sequences)
    print(targets)        
    
    X = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH, padding='pre')
    y = np.array(targets)
    print(X)
    print(y)

    data_to_save = {
        'X': X,
        'y': y,
        'movie_to_idx': movie_to_idx,
        'idx_to_movie': idx_to_movie,
        'n_unique_movies': n_unique_movies
    }
    
    with open('data/sequential_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
        
    print(f"Data prepared and saved to 'data/sequential_data.pkl'. Shape of X: {X.shape}")
if __name__ == "__main__":
    create_sequences()