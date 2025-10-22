import pandas as pd
from sqlalchemy import func, desc
from sqlalchemy.orm import joinedload
from src.Database.database import get_db_session
from src.Database.models import User, Rating, Feedback
import logging

logger = logging.getLogger(__name__)

class AdminManager:
    """
    It handles fetching data for the admin dashboard.
    """

    def get_key_metrics(self):
        """
        key metrics fetched : total users, total ratings, and total feedback.
        """
        try:
            with get_db_session() as session:
                total_users = session.query(func.count(User.id)).scalar()
                total_ratings = session.query(func.count(Rating.id)).scalar()
                total_feedback = session.query(func.count(Feedback.id)).scalar()
                
                return {
                    "total_users": total_users,
                    "total_ratings": total_ratings,
                    "total_feedback": total_feedback,
                }
        except Exception as e:
            logger.error(f"Error fetching key metrics: {e}")
            return None

    def get_most_rated_movies(self, movies_df, limit=10):
        """
        It fetches the most rated movies and joins with movie titles.
        """
        try:
            with get_db_session() as session:
                most_rated = (
                    session.query(
                        Rating.movie_id,
                        func.count(Rating.movie_id).label("rating_count")
                    )
                    .group_by(Rating.movie_id)
                    .order_by(desc("rating_count"))
                    .limit(limit)
                    .all()
                )
                
                if not most_rated:
                    return pd.DataFrame()

                most_rated_df = pd.DataFrame(most_rated, columns=['movie_id', 'rating_count'])
                
                merged_df = pd.merge(
                    most_rated_df,
                    movies_df[['movie_id', 'title']],
                    on='movie_id',
                    how='left'
                )
                return merged_df

        except Exception as e:
            logger.error(f"Error fetching most rated movies: {e}")
            return pd.DataFrame()

    def get_all_feedback(self):
        """
        It retrieves all feedback entries with all needed data fully loaded.
        """
        try:
            with get_db_session() as session:
                feedback_data = (
                    session.query(Feedback)
                    .options(joinedload(Feedback.user))
                    .order_by(desc(Feedback.submitted_at))
                    .all()
                )
                
                feedback_list = [
                    (f.id, f.user.username, f.submitted_at, f.feedback_text)
                    for f in feedback_data
                ]
                return feedback_list
        except Exception as e:
            logger.error(f"Error fetching feedback: {e}")
            return []

    def get_user_activity(self, limit=10):
        """
        It fetches the most active users based on the number of ratings.
        """
        try:
            with get_db_session() as session:
                user_activity = (
                    session.query(
                        User.username,
                        func.count(Rating.id).label("rating_count")
                    )
                    .join(Rating, User.id == Rating.user_id)
                    .group_by(User.username)
                    .order_by(desc("rating_count"))
                    .limit(limit)
                    .all()
                )
                return pd.DataFrame(user_activity, columns=['username', 'ratings_submitted'])
        except Exception as e:
            logger.error(f"Error fetching user activity: {e}")
            return pd.DataFrame()


