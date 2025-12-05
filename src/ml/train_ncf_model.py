import os
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

import mlflow
import mlflow.keras

def train_and_save_ncf_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("ncf_movie_recommender")
    
    print("Loading ratings data...")
    ratings_df = pd.read_csv('data/ratings.csv')
    
    # Create mappings
    user_ids = ratings_df['user_id'].unique().tolist()
    movie_ids = ratings_df['movie_id'].unique().tolist()
    
    user_to_idx = {original_id: i for i, original_id in enumerate(user_ids)}
    movie_to_idx = {original_id: i for i, original_id in enumerate(movie_ids)}
    
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie_to_idx)
    
    scaler = MinMaxScaler()
    ratings_df['rating_scaled'] = scaler.fit_transform(ratings_df['rating'].values.reshape(-1, 1))
    
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)
    
    with mlflow.start_run(run_name="ncf_v1"):
        # Log parameters
        EMBEDDING_SIZE = 50
        BATCH_SIZE = 64
        EPOCHS = 20
        LEARNING_RATE = 0.001
        
        mlflow.log_param("embedding_size", EMBEDDING_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("n_users", n_users)
        mlflow.log_param("n_movies", n_movies)
        
        # Build model (same as before)
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(n_users, EMBEDDING_SIZE, name='user_embedding')(user_input)
        user_vec = Flatten(name='flatten_user_vec')(user_embedding)
        
        movie_input = Input(shape=(1,), name='movie_input')
        movie_embedding = Embedding(n_movies, EMBEDDING_SIZE, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='flatten_movie_vec')(movie_embedding)
        
        concat = Concatenate()([user_vec, movie_vec])
        dense_1 = Dense(128, activation='relu')(concat)
        dense_2 = Dense(64, activation='relu')(dense_1)
        dense_3 = Dense(32, activation='relu')(dense_2)
        output = Dense(1, activation='sigmoid')(dense_3)
        
        model = Model([user_input, movie_input], output)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        # Prepare data
        X = ratings_df[['user_idx', 'movie_idx']].values
        y = ratings_df['rating_scaled'].values
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train_split = [X_train[:, 0], X_train[:, 1]]
        X_val_split = [X_val[:, 0], X_val[:, 1]]
        
        # Custom callback
        class MLflowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metric("train_loss", logs['loss'], step=epoch)
                mlflow.log_metric("train_rmse", logs['root_mean_squared_error'], step=epoch)
                mlflow.log_metric("val_loss", logs['val_loss'], step=epoch)
                mlflow.log_metric("val_rmse", logs['val_root_mean_squared_error'], step=epoch)
        
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('data/ncf_model.keras', save_best_only=True),
            MLflowCallback()
        ]
        
        print("Starting model training...")
        history = model.fit(
            x=X_train_split, 
            y=y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val_split, y_val),
            callbacks=callbacks
        )
        
        # Log final metrics
        mlflow.log_metric("final_train_loss", history.history['loss'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
        
        # Log model
        mlflow.keras.log_model(model, "ncf_model")
        
        # Log artifacts
        mlflow.log_artifact('data/ncf_mappings.pkl')
        
        # Add tags
        mlflow.set_tag("model_type", "NCF")
        mlflow.set_tag("framework", "tensorflow/keras")
        mlflow.set_tag("use_case", "collaborative_filtering")
        
        print("Training complete. Model logged to MLflow.")

if __name__ == "__main__":
    train_and_save_ncf_model()
