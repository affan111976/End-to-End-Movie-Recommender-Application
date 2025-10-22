from sqlalchemy import (Column, Integer, String, DateTime, Boolean, 
                        ForeignKey, Float, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    
    is_admin = Column(Boolean, default=False, nullable=False)

    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    watchlist_items = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")

class Rating(Base):
    __tablename__ = "ratings"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, nullable=False)
    rating = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="ratings")

class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, nullable=False)
    movie_title = Column(String(255), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="watchlist_items")

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    feedback_text = Column(Text, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="feedbacks")
