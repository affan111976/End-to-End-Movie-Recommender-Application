# Movie Recommender System

This is a comprehensive Movie Recommender System built with Python and Streamlit. It provides personalized movie recommendations based on user preferences and features a complete user authentication system, a user dashboard, and an admin panel for monitoring the application.

# Features

# Content-Based Recommendations:
Get movie suggestions based on the movies you like. The recommendation engine analyzes movie metadata like genres, keywords, cast, and crew to find similar movies.
User Authentication: Secure user registration and login system.

# User Dashboard:
Watchlist - Add movies you want to watch later.
Ratings -  Rate movies you've seen to improve future recommendations.
Profile Management - Update your username and email.

# Search & Rate 
Easily search for movies in the database and rate them.

# Admin Dashboard

Key Metrics - View statistics like total users, ratings, and feedback.
Top Movies & Users - See the most rated movies and the most active users.
User Feedback - Review feedback submitted by users.

# Interactive UI
A user-friendly and interactive interface built with Streamlit.

# Tech Stack

Backend - Python
Frontend - Streamlit
Data Manipulation - Pandas, NumPy
Machine Learning - Scikit-learn (for `CountVectorizer` and `cosine_similarity`)
Database - SQLAlchemy with SQLite
API - Requests to fetch movie data from TMDB and to use LLM for text generation and emmbeddings

# Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

# Prerequisites

Python 3.8 or higher
pip

# Installation

1.  Clone the repository
      # bash
    git clone [https://github.com/your-username/movie-recommender-system.git](https://github.com/your-username/movie-recommender-system.git)

2.   Create and activate a virtual environment (recommended)
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    

3.  Install the required dependencies
    pip install -r requirements.txt
    

4.  Get your TMDB API Key 
    Go to [The Movie Database (TMDB)](https://www.themoviedb.org/signup) and create an account.
    Go to your account settings, find the API section, and generate an API key.
    Open the `src/tmdb_utils.py` file and replace `"YOUR_API_KEY"` with your actual TMDB API key.

# Running the Application

1.  Initialize the database and run the Streamlit app
    streamlit run app.py

2.  Open your web browser and go to the local URL provided by Streamlit (usually http://localhost:8501)

##  usage

# Register/Login 
Create a new account or log in 

# Get Recommendations

Select a movie from the dropdown on the sidebar and click "Get Recommendations"

# Explore
 Use the navigation menu on the sidebar to explore different pages like "Search & Rate", "My Dashboard", "Submit Feedback"

# Project Structure

├── data/
│   ├── movie_list.pkl
│   └── similarity.pkl
├── src/
│   ├── admin/
│   │   ├── admin_manager.py
│   │   └── admin_pages.py
│   ├── database/
│   │   ├── database.py
│   │   ├── models.py
│   │   └── user_manager.py
│   ├── recommender.py
│   └── tmdb_utils.py
├── app.py
├── requirements.txt
└── README.md


#  Contributing

Contributions are welcome! If you have suggestions for improvements, please open an issue or submit a pull request.

