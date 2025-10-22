import pandas as pd
# SQLALchemy helps the python programs to communicate with databases
from sqlalchemy import create_engine
import os

def fetch_ratings_data():
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:Affan_111%402004@localhost:5432/movie_recommender_pg")
    try:
        engine = create_engine(db_url)
        query = "SELECT user_id, movie_id, rating, created_at FROM ratings"
        ratings_df = pd.read_sql(query, engine)
        
        ratings_df.to_csv('data/ratings.csv', index=False)
        print(f"Successfully saved {len(ratings_df)} ratings to data/ratings.csv")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    fetch_ratings_data()