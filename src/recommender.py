import pickle
import pandas as pd
import numpy as np
from src.tmdb_utils import fetch_poster, fetch_movie_details
# load_model function used to load a pre trained model and saved machine learning model
from tensorflow.keras.models import load_model 

# Loading existing files from the data folder
movies = pickle.load(open('data/movie_list.pkl', 'rb'))
similarity = pickle.load(open('data/similarity.pkl', 'rb'))

from tensorflow.keras.preprocessing.sequence import pad_sequences
from .ml.prepare_sequential_data import SEQUENCE_LENGTH 

# Load RNN Model and Mappings 
try:
    rnn_model = load_model('data/rnn_model.keras')
    with open('data/rnn_mappings.pkl', 'rb') as f:
        rnn_mappings = pickle.load(f)
    rnn_movie_to_idx = rnn_mappings['movie_to_idx']
    print(rnn_movie_to_idx)
    rnn_idx_to_movie = rnn_mappings['idx_to_movie']
    ratings_df_full = pd.read_csv('data/ratings.csv', parse_dates=['created_at'])
    print("RNN model loaded successfully.")
except Exception as e:
    print(f"Error loading RNN model: {e}. Sequential recommendations will be disabled.")
    rnn_model = None

# Load NCF Model and Mappings 
try:
    ncf_model = load_model('data/ncf_model.keras')
    with open('data/ncf_mappings.pkl', 'rb') as f:
        ncf_mappings = pickle.load(f)
    user_to_idx = ncf_mappings['user_to_idx']
    movie_to_idx = ncf_mappings['movie_to_idx']
    idx_to_movie = ncf_mappings['idx_to_movie']
    print("NCF model loaded successfully.")
except Exception as e:
    print(f"Error loading NCF model: {e}. Collaborative filtering will be disabled.")
    ncf_model = None

def get_rnn_recommendations(user_id, n=5):
    """
    Generates 'what to watch next' recommendations using the RNN model.
    """
    if not rnn_model:
        return []

    try:
        # Getting the user's recent movie history
        user_history_df = ratings_df_full[ratings_df_full['user_id'] == user_id].sort_values('created_at', ascending=False)
        
        if user_history_df.empty:
            return []
            
        # Getting the last SEQUENCE_LENGTH rated movies
        recent_movie_ids = user_history_df['movie_id'].head(SEQUENCE_LENGTH).tolist()
        print(recent_movie_ids)
        
        input_sequence_indices = [rnn_movie_to_idx.get(mid) for mid in recent_movie_ids if mid in rnn_movie_to_idx]
        print(input_sequence_indices)
        padded_sequence = pad_sequences([input_sequence_indices], maxlen=SEQUENCE_LENGTH, padding='pre')
        print(padded_sequence)
        if padded_sequence.shape[1] == 0:
            return []

        # Predict the probabilities for the next movie
        predicted_probabilities = rnn_model.predict(padded_sequence)[0]
        
        # Get top N movie indices, ignoring the padding token (index 0)
        top_indices = predicted_probabilities.argsort()[- (n + len(recent_movie_ids)) :][::-1]
        
        # Filter out already-seen movies and convert back to movie IDs
        recommended_ids = []
        for idx in top_indices:
            if idx > 0 and rnn_idx_to_movie[idx] not in recent_movie_ids:
                recommended_ids.append(rnn_idx_to_movie[idx])
            if len(recommended_ids) >= n:
                break
        
        # Fetch details for the recommended movies
        recs = []
        for movie_id in recommended_ids:
            movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
            recs.append({
                'title': movie_info['title'],
                'poster': fetch_poster(movie_id),
                'details': fetch_movie_details(movie_id)
            })
        return recs
        
    except Exception as e:
        print(f"Error during RNN recommendation: {e}")
        return []

# For content based recommendations
def get_content_based_recommendations(movie_title, n=20):
    try:
        index = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return []
    
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_ids = [int(movies.iloc[i[0]].movie_id) for i in distances[1:n+1]]
    return recommended_movie_ids

# NCF Recommendation Function 
def get_ncf_recommendations(user_id, n=20):
    """
    Generates N movie recommendations for a user using the trained NCF model.
    """
    if not ncf_model or user_id not in user_to_idx:
        print(f"User {user_id} not in the NCF model or model not loaded.")
        return []

    user_idx = user_to_idx[user_id]
    
    # Find movies the user has NOT rated
    rated_movie_ids = movies[movies['movie_id'].isin(
        pd.read_csv('data/ratings.csv').query(f'user_id == "{user_id}"')['movie_id']
    )]
    all_movie_ids = set(movie_to_idx.keys())
    rated_movie_ids_set = set(rated_movie_ids['movie_id'])
    unrated_movie_ids = list(all_movie_ids - rated_movie_ids_set)
    
    # Map unrated movie IDs to their indices
    unrated_movie_indices = [movie_to_idx[movie_id] for movie_id in unrated_movie_ids]

    # If there are no movies to recommend (user has rated them all), return early.
    if not unrated_movie_indices:
        return []

    # Prepare model inputs
    user_input_array = np.full(len(unrated_movie_indices), user_idx)
    movie_input_array = np.array(unrated_movie_indices)

    # Predict scores
    predictions = ncf_model.predict([user_input_array, movie_input_array], verbose=0).flatten()
    
    # Get top N recommendations
    top_indices = predictions.argsort()[-n:][::-1]
    recommended_movie_indices = [unrated_movie_indices[i] for i in top_indices]
    
    # Convert indices back to original movie IDs
    recommended_movie_ids = [idx_to_movie[i] for i in recommended_movie_indices]

    return recommended_movie_ids

# Hybrid Recommendation Function
def get_hybrid_recommendations(user_id, movie_title, n=5):
    content_recs = get_content_based_recommendations(movie_title, n=20)
    
    # Use the new NCF function for collaborative filtering
    collaborative_recs = get_ncf_recommendations(user_id, n=20) 
    
    hybrid_scores = {}
    for movie_id in collaborative_recs:
        hybrid_scores[movie_id] = hybrid_scores.get(movie_id, 0) + 1.5 
        
    for movie_id in content_recs:
        hybrid_scores[movie_id] = hybrid_scores.get(movie_id, 0) + 1.0
    print(hybrid_scores)
    sorted_recs = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
    print(sorted_recs)
    
    final_recommendations_ids = []
    input_movie_id = movies[movies['title'] == movie_title]['movie_id'].iloc[0]
    for movie_id, score in sorted_recs:
        if movie_id != input_movie_id:
            final_recommendations_ids.append(movie_id)
        if len(final_recommendations_ids) >= n:
            break
            
    recommended_movies_details = []
    for movie_id in final_recommendations_ids:
        movie_info_df = movies[movies['movie_id'] == movie_id]
        if not movie_info_df.empty:
            movie_info = movie_info_df.iloc[0]
            recommended_movies_details.append({
                'title': movie_info['title'],
                'poster': fetch_poster(movie_id),
                'details': fetch_movie_details(movie_id)
            })
        
    return recommended_movies_details

