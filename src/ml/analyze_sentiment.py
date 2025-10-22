import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def analyze_and_save_sentiments():
    """
    Analyzes sentiments from the local sample review file and saves aggregated scores.
    """
    print("Loading pre-trained sentiment analysis model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    print("Loading sample review data from 'data/reviews_sample.csv'...")
    try:
        reviews_df = pd.read_csv('data/reviews_sample.csv')
    except FileNotFoundError:
        print("Error: 'data/reviews_sample.csv' not found.")
        print("Please create this file in the 'data' folder with the sample content.")
        return

    # Group reviews by each movie
    reviews_by_movie = reviews_df.groupby('movie_id')

    movie_sentiments = []
    print("Analyzing sentiments for each movie in the sample...")
    
    for movie_id, group in tqdm(reviews_by_movie, desc="Analyzing Movies"):
        reviews = group['review_text'].tolist()
        movie_title = group['title'].iloc[0]

        # Analyze sentiments for all reviews of the movie
        results = sentiment_pipeline(reviews, truncation=True, max_length=512)
        
        # Aggregate the scores
        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        positivity_ratio = positive_count / len(results) if results else 0
        
        movie_sentiments.append({
            'movie_id': movie_id,
            'title': movie_title,
            'positivity_ratio': positivity_ratio,
            'review_count': len(reviews)
        })

    # Save the aggregated results
    sentiments_df = pd.DataFrame(movie_sentiments)
    output_path = 'data/movie_sentiments.csv'
    sentiments_df.to_csv(output_path, index=False)
    
    print(f"\nSentiment analysis complete. Saved {len(sentiments_df)} movie sentiment records to '{output_path}'")
    print(sentiments_df.head())

if __name__ == "__main__":
    analyze_and_save_sentiments()