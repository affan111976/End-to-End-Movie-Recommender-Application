import requests
import streamlit as st

API_KEY = "57f538d6f476644e2c695c112870b186"
BASE_URL = "https://api.themoviedb.org/3/movie/{}"

def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB."""
    try:
        url = BASE_URL.format(movie_id) + f"?api_key={API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except Exception as e:
        print(e)
        return "https://via.placeholder.com/500x750?text=Error"

def fetch_movie_details(movie_id):
    """Fetches additional movie details like overview, rating, and trailer."""
    try:
        url = BASE_URL.format(movie_id) + f"?api_key={API_KEY}&language=en-US&append_to_response=videos"
        data = requests.get(url).json()

        details = {
            'overview': data.get('overview', 'No overview available.'),
            'vote_average': data.get('vote_average', 0),
            'release_date': data.get('release_date', 'N/A'),
            'trailer_key': None
        }

        # Finding the official trailer from the videos response
        if 'videos' in data and data['videos']['results']:
            for video in data['videos']['results']:
                if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                    details['trailer_key'] = video['key']
                    break
        return details
    except Exception as e:
        print(e)
        return {'overview': 'Error fetching details.', 'vote_average': 0, 'release_date': 'N/A', 'trailer_key': None}