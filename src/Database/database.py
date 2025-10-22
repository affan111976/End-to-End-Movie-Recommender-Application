import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_url: str = None):
        if database_url is None:
           database_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:Affan_111%402004@localhost:5432/movie_recommender_pg")
        
        if database_url.startswith("sqlite"):
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                },
                echo=False  
            )
        else:
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Getting database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """Getting database session for direct use (but it will be closed!)"""
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """Checking if database is accessible"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Global database instance
db_manager = DatabaseManager()

def init_database():
    """Initializing database with tables"""
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
        return True  
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False 


def get_db_session():
    """Dependency function for getting database session"""
    return db_manager.get_session()

# Testing database connection
if __name__ == "__main__":
    print("Testing database connection...")
    db_manager = DatabaseManager()
    
    if db_manager.health_check():
        print("✅ Database connection successful")
        db_manager.create_tables()
        print("✅ Tables created successfully")
    else:
        print("❌ Database connection failed")