import mlflow
import mlflow.keras
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_rnn():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("rnn_movie_recommender")
    
    print("Loading prepared sequential data...")
    with open('data/sequential_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    n_unique_movies = data['n_unique_movies']
    
    # Start MLflow run
    with mlflow.start_run(run_name="lstm_v1"):
        # Log parameters
        EMBEDDING_DIM = 64
        LSTM_UNITS = 128
        BATCH_SIZE = 128
        EPOCHS = 30
        
        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("lstm_units", LSTM_UNITS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("sequence_length", X.shape[1])
        mlflow.log_param("n_unique_movies", n_unique_movies)
        
        # Build model
        model = Sequential([
            Embedding(input_dim=n_unique_movies, output_dim=EMBEDDING_DIM, mask_zero=True),
            Dropout(0.2),
            LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
            Dropout(0.2),
            Dense(n_unique_movies, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Custom callback to log metrics
        class MLflowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metric("train_loss", logs['loss'], step=epoch)
                mlflow.log_metric("train_accuracy", logs['accuracy'], step=epoch)
                mlflow.log_metric("val_loss", logs['val_loss'], step=epoch)
                mlflow.log_metric("val_accuracy", logs['val_accuracy'], step=epoch)
        
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('data/rnn_model.keras', save_best_only=True),
            MLflowCallback()
        ]
        
        print("\nStarting RNN model training...")
        history = model.fit(
            X, y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=callbacks
        )
        
        # Log final metrics
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        
        # Log model
        mlflow.keras.log_model(model, "rnn_model")
        
        # Log artifacts
        mlflow.log_artifact('data/rnn_mappings.pkl')
        
        # Add tags
        mlflow.set_tag("model_type", "LSTM")
        mlflow.set_tag("framework", "tensorflow/keras")
        mlflow.set_tag("use_case", "sequential_recommendation")
        
        print("\nTraining complete. Model logged to MLflow.")

if __name__ == "__main__":
    train_rnn()