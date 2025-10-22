import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_rnn():
    print("Loading prepared sequential data...")
    with open('data/sequential_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data)
    print(type(data))

    X = data['X']
    y = data['y']
    n_unique_movies = data['n_unique_movies']
    
    # Build the LSTM Model 
    EMBEDDING_DIM = 64
    
    model = Sequential([
        # The Embedding layer turns movie indices into dense vectors of a fixed size.
        # mask_zero=True tells the model to ignore the padding '0's.
        Embedding(input_dim=n_unique_movies, output_dim=EMBEDDING_DIM, mask_zero=True),
        
        # The LSTM layer processes the sequence of embeddings.
        LSTM(128),
        
        # The final Dense layer outputs a probability distribution over all movies.
        Dense(n_unique_movies, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary()

    #  Train the Model 
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('data/rnn_model.keras', save_best_only=True)
    ]
    
    print("\nStarting RNN model training...")
    model.fit(
        X, y,
        batch_size=128,
        epochs=30,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    # Save the mappings separately
    mappings = {
        'movie_to_idx': data['movie_to_idx'],
        'idx_to_movie': data['idx_to_movie']
    }
    with open('data/rnn_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
        
    print("\nTraining complete. Model and mappings saved.")
    
if __name__ == "__main__":
    train_rnn()