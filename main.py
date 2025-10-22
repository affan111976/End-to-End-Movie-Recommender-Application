from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import jwt
from datetime import datetime, timedelta
import os
from sqlalchemy.orm import Session

from src.Database.database import get_db_session, init_database
from src.Database.user_manager import UserManager
from src.Database.models import User, Rating, WatchlistItem, Feedback
from src.recommender import (
    get_hybrid_recommendations,
    get_rnn_recommendations,
    get_ncf_recommendations,
    movies
)
from src.tmdb_utils import fetch_poster, fetch_movie_details

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="Backend API for Movie Recommender System",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Security
security = HTTPBearer()

# Initialize database and user manager
user_manager = UserManager()

# Pydantic Models

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    terms_consent: bool = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str
    is_admin: bool

class UserProfile(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime
    is_admin: bool

class UserProfileUpdate(BaseModel):
    new_username: Optional[str] = None
    new_email: Optional[EmailStr] = None

class RatingCreate(BaseModel):
    movie_id: int
    rating: float

class RatingResponse(BaseModel):
    id: str
    user_id: str
    movie_id: int
    rating: float
    created_at: datetime
    movie_title: Optional[str] = None

class WatchlistAdd(BaseModel):
    movie_id: int
    movie_title: str

class WatchlistResponse(BaseModel):
    id: str
    movie_id: int
    movie_title: str
    added_at: datetime

class FeedbackCreate(BaseModel):
    feedback_text: str

class MovieRecommendation(BaseModel):
    title: str
    poster: str
    details: dict

class GenreMovies(BaseModel):
    genre: str
    limit: int = 10

# JWT Helper Functions

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Startup Event

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    if not init_database():
        raise Exception("Failed to initialize database")
    user_manager.ensure_admin_exists()
    print("✅ Database initialized and admin user ensured")

# Health Check

@app.get("/")
async def root():
    return {
        "message": "Movie Recommender API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Authentication Endpoints

@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    """Register a new user"""
    if not user_data.terms_consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must accept the terms of service"
        )
    
    user = user_manager.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user['id'], "username": user['username']}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user['id'],
        "username": user['username'],
        "is_admin": user['is_admin']
    }

@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login user and return JWT token"""
    user = user_manager.authenticate_user(
        username=credentials.username,
        password=credentials.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user['id'], "username": user['username']}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user['id'],
        "username": user['username'],
        "is_admin": user['is_admin']
    }

# User Profile Endpoints 

@app.get("/api/user/profile", response_model=UserProfile)
async def get_profile(user_id: str = Depends(verify_token)):
    """Get user profile"""
    try:
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.put("/api/user/profile")
async def update_profile(
    profile_data: UserProfileUpdate,
    user_id: str = Depends(verify_token)
):
    """Update user profile"""
    success, message = user_manager.update_user_profile(
        user_id=user_id,
        new_username=profile_data.new_username,
        new_email=profile_data.new_email
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return {"message": message}

# Recommendations Endpoints 

@app.get("/api/recommendations/rnn", response_model=List[MovieRecommendation])
async def get_rnn_recs(
    n: int = 5,
    user_id: str = Depends(verify_token)
):
    """Get RNN-based sequential recommendations"""
    recommendations = get_rnn_recommendations(user_id=user_id, n=n)
    return recommendations

@app.get("/api/recommendations/hybrid", response_model=List[MovieRecommendation])
async def get_hybrid_recs(
    movie_title: str,
    n: int = 5,
    user_id: str = Depends(verify_token)
):
    """Get hybrid recommendations for a specific movie"""
    recommendations = get_hybrid_recommendations(
        user_id=user_id,
        movie_title=movie_title,
        n=n
    )
    
    if not recommendations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendations found for this movie"
        )
    
    return recommendations

@app.get("/api/recommendations/ncf", response_model=List[int])
async def get_ncf_recs(
    n: int = 20,
    user_id: str = Depends(verify_token)
):
    """Get NCF-based collaborative filtering recommendations"""
    recommendations = get_ncf_recommendations(user_id=user_id, n=n)
    return recommendations

# Movies Endpoints 

@app.get("/api/movies/search")
async def search_movies(
    query: str,
    limit: int = 20,
    user_id: str = Depends(verify_token)
):
    """Search for movies by title"""
    results = movies[movies['title'].str.contains(query, case=False, na=False)]
    
    if results.empty:
        return []
    
    movie_list = []
    for _, row in results.head(limit).iterrows():
        movie_list.append({
            "movie_id": int(row['movie_id']),
            "title": row['title'],
            "poster": fetch_poster(row['movie_id'])
        })
    
    return movie_list

@app.get("/api/movies/genre/{genre}")
async def get_movies_by_genre(
    genre: str,
    limit: int = 10,
    user_id: str = Depends(verify_token)
):
    """Get movies by genre"""
    genre_movies = movies[movies['genres'].apply(lambda x: genre in x)].head(limit)
    
    if genre_movies.empty:
        return []
    
    movie_list = []
    for _, row in genre_movies.iterrows():
        movie_list.append({
            "movie_id": int(row['movie_id']),
            "title": row['title'],
            "poster": fetch_poster(row['movie_id']),
            "genres": row['genres']
        })
    
    return movie_list

@app.get("/api/movies/{movie_id}")
async def get_movie_details_endpoint(
    movie_id: int,
    user_id: str = Depends(verify_token)
):
    """Get detailed information about a specific movie"""
    details = fetch_movie_details(movie_id)
    poster = fetch_poster(movie_id)
    
    movie_info = movies[movies['movie_id'] == movie_id]
    if not movie_info.empty:
        title = movie_info.iloc[0]['title']
    else:
        title = "Unknown"
    
    return {
        "movie_id": movie_id,
        "title": title,
        "poster": poster,
        "details": details
    }

# Ratings Endpoints 

@app.post("/api/ratings", response_model=RatingResponse)
async def create_rating(
    rating_data: RatingCreate,
    user_id: str = Depends(verify_token)
):
    """Create or update a movie rating"""
    try:
        with get_db_session() as session:
            # Check if rating already exists
            existing_rating = session.query(Rating).filter_by(
                user_id=user_id,
                movie_id=rating_data.movie_id
            ).first()
            
            if existing_rating:
                # Update existing rating
                existing_rating.rating = rating_data.rating
                session.commit()
                rating_obj = existing_rating
            else:
                # Create new rating
                new_rating = Rating(
                    user_id=user_id,
                    movie_id=rating_data.movie_id,
                    rating=rating_data.rating
                )
                session.add(new_rating)
                session.commit()
                rating_obj = new_rating
            
            # Get movie title
            movie_info = movies[movies['movie_id'] == rating_data.movie_id]
            movie_title = movie_info.iloc[0]['title'] if not movie_info.empty else None
            
            return {
                "id": rating_obj.id,
                "user_id": rating_obj.user_id,
                "movie_id": rating_obj.movie_id,
                "rating": rating_obj.rating,
                "created_at": rating_obj.created_at,
                "movie_title": movie_title
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/ratings", response_model=List[RatingResponse])
async def get_user_ratings(user_id: str = Depends(verify_token)):
    """Get all ratings for the current user"""
    try:
        with get_db_session() as session:
            ratings = session.query(Rating).filter_by(
                user_id=user_id
            ).order_by(Rating.created_at.desc()).all()
            
            result = []
            for rating in ratings:
                movie_info = movies[movies['movie_id'] == rating.movie_id]
                movie_title = movie_info.iloc[0]['title'] if not movie_info.empty else None
                
                result.append({
                    "id": rating.id,
                    "user_id": rating.user_id,
                    "movie_id": rating.movie_id,
                    "rating": rating.rating,
                    "created_at": rating.created_at,
                    "movie_title": movie_title
                })
            
            return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/api/ratings/{rating_id}")
async def delete_rating(
    rating_id: str,
    user_id: str = Depends(verify_token)
):
    """Delete a rating"""
    try:
        with get_db_session() as session:
            rating = session.query(Rating).filter_by(
                id=rating_id,
                user_id=user_id
            ).first()
            
            if not rating:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Rating not found"
                )
            
            session.delete(rating)
            session.commit()
            
            return {"message": "Rating deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Watchlist Endpoints 

@app.post("/api/watchlist", response_model=WatchlistResponse)
async def add_to_watchlist(
    watchlist_data: WatchlistAdd,
    user_id: str = Depends(verify_token)
):
    """Add a movie to watchlist"""
    try:
        with get_db_session() as session:
            # Check if already in watchlist
            existing = session.query(WatchlistItem).filter_by(
                user_id=user_id,
                movie_id=watchlist_data.movie_id
            ).first()
            
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Movie already in watchlist"
                )
            
            new_item = WatchlistItem(
                user_id=user_id,
                movie_id=watchlist_data.movie_id,
                movie_title=watchlist_data.movie_title
            )
            session.add(new_item)
            session.commit()
            
            return {
                "id": new_item.id,
                "movie_id": new_item.movie_id,
                "movie_title": new_item.movie_title,
                "added_at": new_item.added_at
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/watchlist", response_model=List[WatchlistResponse])
async def get_watchlist(user_id: str = Depends(verify_token)):
    """Get user's watchlist"""
    try:
        with get_db_session() as session:
            items = session.query(WatchlistItem).filter_by(
                user_id=user_id
            ).order_by(WatchlistItem.added_at.desc()).all()
            
            return [
                {
                    "id": item.id,
                    "movie_id": item.movie_id,
                    "movie_title": item.movie_title,
                    "added_at": item.added_at
                }
                for item in items
            ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/api/watchlist/{item_id}")
async def remove_from_watchlist(
    item_id: str,
    user_id: str = Depends(verify_token)
):
    """Remove a movie from watchlist"""
    try:
        with get_db_session() as session:
            item = session.query(WatchlistItem).filter_by(
                id=item_id,
                user_id=user_id
            ).first()
            
            if not item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Watchlist item not found"
                )
            
            session.delete(item)
            session.commit()
            
            return {"message": "Removed from watchlist"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Feedback Endpoints

@app.post("/api/feedback")
async def submit_feedback(
    feedback_data: FeedbackCreate,
    user_id: str = Depends(verify_token)
):
    """Submit user feedback"""
    if not feedback_data.feedback_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback text cannot be empty"
        )
    
    success = user_manager.submit_feedback(
        user_id=user_id,
        feedback_text=feedback_data.feedback_text
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )
    
    return {"message": "Feedback submitted successfully"}

@app.get("/api/feedback")
async def get_all_feedback(user_id: str = Depends(verify_token)):
    """Get all feedback (admin only)"""
    try:
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user or not user.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            feedbacks = session.query(Feedback).order_by(
                Feedback.submitted_at.desc()
            ).all()
            
            return [
                {
                    "id": f.id,
                    "user_id": f.user_id,
                    "username": f.user.username,
                    "feedback_text": f.feedback_text,
                    "submitted_at": f.submitted_at
                }
                for f in feedbacks
            ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Admin Endpoints

@app.get("/api/admin/stats")
async def get_admin_stats(user_id: str = Depends(verify_token)):
    """Get admin statistics"""
    try:
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user or not user.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            from sqlalchemy import func
            from src.Database.admin_manager import AdminManager
            
            admin_manager = AdminManager()
            metrics = admin_manager.get_key_metrics()
            most_rated = admin_manager.get_most_rated_movies(movies, limit=10)
            user_activity = admin_manager.get_user_activity(limit=10)
            
            return {
                "metrics": metrics,
                "most_rated_movies": most_rated.to_dict('records') if not most_rated.empty else [],
                "top_users": user_activity.to_dict('records') if not user_activity.empty else []
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)