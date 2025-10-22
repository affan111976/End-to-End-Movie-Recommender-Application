import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_and_save_ncf_model():
    print("Loading ratings data...")
    ratings_df = pd.read_csv('data/ratings.csv')
    
    # Create continuous integer IDs for users and movies
    user_ids = ratings_df['user_id'].unique().tolist()
    movie_ids = ratings_df['movie_id'].unique().tolist()
    
    user_to_idx = {original_id: i for i, original_id in enumerate(user_ids)}
    movie_to_idx = {original_id: i for i, original_id in enumerate(movie_ids)}
    print(user_to_idx)
    print(movie_to_idx)
    
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie_to_idx)
    print(ratings_df.head())

    # Scale ratings to a [0, 1] range for the sigmoid output layer
    scaler = MinMaxScaler()
    ratings_df['rating_scaled'] = scaler.fit_transform(ratings_df['rating'].values.reshape(-1, 1))

    # Get number of unique users and movies for embedding layer sizes
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)
    
    print(f"Found {n_users} unique users and {n_movies} unique movies.")
    
    EMBEDDING_SIZE = 50 

    # User input and embedding
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(n_users, EMBEDDING_SIZE, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_user_vec')(user_embedding)

    # Movie input and embedding
    movie_input = Input(shape=(1,), name='movie_input')
    movie_embedding = Embedding(n_movies, EMBEDDING_SIZE, name='movie_embedding')(movie_input)
    movie_vec = Flatten(name='flatten_movie_vec')(movie_embedding)

    # Concatenate the flattened embedding vectors
    concat = Concatenate()([user_vec, movie_vec])

    # MLP (Multi-Layer Perceptron) tower for learning interactions
    dense_1 = Dense(128, activation='relu')(concat)
    dense_2 = Dense(64, activation='relu')(dense_1)
    dense_3 = Dense(32, activation='relu')(dense_2)
    
    # Output layer with sigmoid activation for a [0, 1] prediction
    output = Dense(1, activation='sigmoid')(dense_3)

    model = Model([user_input, movie_input], output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    model.summary()

    # Train the Model 
    
    X = ratings_df[['user_idx', 'movie_idx']].values
    y = ratings_df['rating_scaled'].values
    print(X)
    print(y)
    print(len(X))
    print(len(y))
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print(len(X_train))
    print(len(X_val))

    X_train_split = [X_train[:, 0], X_train[:, 1]]
    X_val_split = [X_val[:, 0], X_val[:, 1]]
    
    # Callbacks for robust training
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('data/ncf_model.keras', save_best_only=True)

    print("Starting model training...")
    model.fit(
        x=X_train_split, 
        y=y_train,
        batch_size=64,
        epochs=20,
        validation_data=(X_val_split, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    print("Training complete.")

    # Save Mappings 
    
    mappings = {
        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx,
        'idx_to_movie': {i: original_id for original_id, i in movie_to_idx.items()} # For easy lookup later
    }
    with open('data/ncf_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
        
    print("NCF model and mappings saved successfully.")
    
if __name__ == "__main__":
    train_and_save_ncf_model()