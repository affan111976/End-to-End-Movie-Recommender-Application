import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import and_, or_
from .models import User, Feedback
from .database import get_db_session
import logging
import bcrypt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self):
        pass

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        return pwd_context.verify(password, hashed_password)

    def _user_to_dict(self, user: User) -> Optional[Dict[str, Any]]:
        if not user:
            return None
        return {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'created_at': user.created_at,
            'updated_at': user.updated_at,
            'is_active': user.is_active,
            'last_login': user.last_login,
            'is_admin': user.is_admin,
        }

    def create_user(self, username: str, email: str, password: str,
                   first_name: str = None, last_name: str = None, is_admin: bool = False) -> Optional[Dict[str, Any]]:
        try:
            with get_db_session() as session:
                existing_user = session.query(User).filter(
                    or_(User.username == username, User.email == email)
                ).first()
                if existing_user:
                    logger.warning(f"User with username {username} or email {email} already exists")
                    return None

                new_user = User(
                    username=username,
                    email=email,
                    password_hash=self.hash_password(password),
                    first_name=first_name,
                    last_name=last_name,
                    is_admin=is_admin
                )
                session.add(new_user)
                session.commit()
                return self._user_to_dict(new_user)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
            
    def ensure_admin_exists(self):
        """
        Checks if an admin user exists and creates one if not.
        This should be called once at application startup.
        """
        try:
            with get_db_session() as session:
                admin_user = session.query(User).filter(User.is_admin == True).first()
                if not admin_user:
                    logger.info("No admin user found. Creating default admin.")
                    self.create_user(
                        username="admin",
                        email="admin@example.com",
                        password="adminpassword", 
                        is_admin=True
                    )
                    logger.info("Default admin created. Username: 'admin', Password: 'adminpassword'")
        except Exception as e:
            logger.error(f"Error ensuring admin exists: {e}")


    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        try:
            with get_db_session() as session:
                user = session.query(User).filter(
                    and_(
                        or_(User.username == username, User.email == username),
                        User.is_active == True
                    )
                ).first()

                if user and self.verify_password(password, user.password_hash):
                    user.last_login = datetime.utcnow()
                    session.commit()
                    return self._user_to_dict(user)

                logger.warning(f"Authentication failed for user {username}")
                return None
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
            
    def update_user_profile(self, user_id: str, new_username: str = None, new_email: str = None) -> (bool, str):
        try:
            with get_db_session() as session:
                user = session.query(User).get(user_id)
                if not user:
                    return False, "User not found."

                if new_username and new_username != user.username:
                    if session.query(User).filter(User.username == new_username).first():
                        return False, "Username already taken."
                    user.username = new_username

                if new_email and new_email != user.email:
                    if session.query(User).filter(User.email == new_email).first():
                        return False, "Email already in use."
                    user.email = new_email
                
                session.commit()
                return True, "Profile updated successfully!"
        except Exception as e:
            logger.error(f"Error updating profile for user {user_id}: {e}")
            return False, "An error occurred during update."

    def submit_feedback(self, user_id: str, feedback_text: str) -> bool:
        if not feedback_text:
            return False
        try:
            with get_db_session() as session:
                new_feedback = Feedback(
                    user_id=user_id,
                    feedback_text=feedback_text
                )
                session.add(new_feedback)
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error submitting feedback for user {user_id}: {e}")
            return False
