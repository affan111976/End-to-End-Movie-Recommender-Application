import requests
from typing import Optional, Dict, Any, List
import streamlit as st

class APIClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        
    def set_token(self, token: str):
        """Set the authentication token"""
        self.token = token
        
    def get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    # Authentication Methods 
    
    def register(self, username: str, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Register a new user"""
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password,
                    "terms_consent": True
                }
            )
            response.raise_for_status()
            data = response.json()
            self.token = data['access_token']
            return data
        except requests.exceptions.RequestException as e:
            print(f"Registration error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login and get authentication token"""
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data['access_token']
            return data
        except requests.exceptions.RequestException as e:
            print(f"Login error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    # User Profile Methods 
    
    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        try:
            response = requests.get(
                f"{self.base_url}/api/user/profile",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get profile error: {e}")
            return None
    
    def update_profile(self, new_username: Optional[str] = None, 
                      new_email: Optional[str] = None) -> tuple[bool, str]:
        """Update user profile"""
        try:
            response = requests.put(
                f"{self.base_url}/api/user/profile",
                headers=self.get_headers(),
                json={
                    "new_username": new_username,
                    "new_email": new_email
                }
            )
            response.raise_for_status()
            return True, response.json()['message']
        except requests.exceptions.RequestException as e:
            error_msg = "Failed to update profile"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_msg = e.response.json().get('detail', error_msg)
                except:
                    pass
            return False, error_msg
    
    # Recommendations Methods 
    
    def get_rnn_recommendations(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get RNN-based sequential recommendations"""
        try:
            response = requests.get(
                f"{self.base_url}/api/recommendations/rnn",
                headers=self.get_headers(),
                params={"n": n}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"RNN recommendations error: {e}")
            return []
    
    def get_hybrid_recommendations(self, movie_title: str, n: int = 5) -> List[Dict[str, Any]]:
        """Get hybrid recommendations for a movie"""
        try:
            response = requests.get(
                f"{self.base_url}/api/recommendations/hybrid",
                headers=self.get_headers(),
                params={"movie_title": movie_title, "n": n}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Hybrid recommendations error: {e}")
            return []
    
    # Movies Methods 
    
    def search_movies(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for movies"""
        try:
            response = requests.get(
                f"{self.base_url}/api/movies/search",
                headers=self.get_headers(),
                params={"query": query, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Search movies error: {e}")
            return []
    
    def get_movies_by_genre(self, genre: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get movies by genre"""
        try:
            response = requests.get(
                f"{self.base_url}/api/movies/genre/{genre}",
                headers=self.get_headers(),
                params={"limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get movies by genre error: {e}")
            return []
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Get movie details"""
        try:
            response = requests.get(
                f"{self.base_url}/api/movies/{movie_id}",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get movie details error: {e}")
            return None
    
    # Ratings Methods 
    
    def create_rating(self, movie_id: int, rating: float) -> Optional[Dict[str, Any]]:
        """Create or update a rating"""
        try:
            response = requests.post(
                f"{self.base_url}/api/ratings",
                headers=self.get_headers(),
                json={"movie_id": movie_id, "rating": rating}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Create rating error: {e}")
            return None
    
    def get_ratings(self) -> List[Dict[str, Any]]:
        """Get all user ratings"""
        try:
            response = requests.get(
                f"{self.base_url}/api/ratings",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get ratings error: {e}")
            return []
    
    def delete_rating(self, rating_id: str) -> bool:
        """Delete a rating"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/ratings/{rating_id}",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Delete rating error: {e}")
            return False
    
    # Watchlist Methods 
    
    def add_to_watchlist(self, movie_id: int, movie_title: str) -> Optional[Dict[str, Any]]:
        """Add movie to watchlist"""
        try:
            response = requests.post(
                f"{self.base_url}/api/watchlist",
                headers=self.get_headers(),
                json={"movie_id": movie_id, "movie_title": movie_title}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Add to watchlist error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get('detail', str(e))
                    print(f"Error detail: {error_detail}")
                except:
                    pass
            return None
    
    def get_watchlist(self) -> List[Dict[str, Any]]:
        """Get user watchlist"""
        try:
            response = requests.get(
                f"{self.base_url}/api/watchlist",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get watchlist error: {e}")
            return []
    
    def remove_from_watchlist(self, item_id: str) -> bool:
        """Remove movie from watchlist"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/watchlist/{item_id}",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Remove from watchlist error: {e}")
            return False
     
    # Feedback Methods 
    
    def submit_feedback(self, feedback_text: str) -> bool:
        """Submit user feedback"""
        try:
            response = requests.post(
                f"{self.base_url}/api/feedback",
                headers=self.get_headers(),
                json={"feedback_text": feedback_text}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Submit feedback error: {e}")
            return False
    
    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Get all feedback (admin only)"""
        try:
            response = requests.get(
                f"{self.base_url}/api/feedback",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get feedback error: {e}")
            return []
    
    # Admin Methods 
    
    def get_admin_stats(self) -> Optional[Dict[str, Any]]:
        """Get admin statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/api/admin/stats",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get admin stats error: {e}")
            return None
    
    # Health Check
    
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False