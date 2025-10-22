# frontend
import streamlit as st
import pandas as pd
import logging
from api_client import APIClient
from src.recommender import movies
from src.tmdb_utils import fetch_poster

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API Client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient(base_url="http://localhost:8000")

api_client = st.session_state.api_client

# Check API health
if 'api_health_checked' not in st.session_state:
    if not api_client.health_check():
        st.error("⚠️ Cannot connect to the backend API. Please make sure the FastAPI server is running on http://localhost:8000")
        st.info("Run: `uvicorn main:app --reload` to start the backend server")
        st.stop()
    st.session_state.api_health_checked = True

def login_page():
    st.markdown('<h1 class="main-header">🎬 Movie Recommender - Login</h1>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username and password:
                    try:
                        user_data = api_client.login(username, password)
                        if user_data:
                            st.session_state.user_id = user_data['user_id']
                            st.session_state.username = user_data['username']
                            st.session_state.is_admin = user_data['is_admin']
                            st.session_state.access_token = user_data['access_token']
                            st.session_state.logged_in = True
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    except Exception as e:
                        st.error(f"Login error: {str(e)}")
                        logger.error(f"Login error for user {username}: {e}")
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            terms_consent = st.checkbox("I agree to the terms of service and privacy policy")
            reg_submitted = st.form_submit_button("Register")
            
            if reg_submitted:
                if reg_username and reg_email and reg_password and terms_consent:
                    try:
                        user_data = api_client.register(reg_username, reg_email, reg_password)
                        if user_data:
                            st.success("Account created successfully! You are now logged in.")
                            st.session_state.user_id = user_data['user_id']
                            st.session_state.username = user_data['username']
                            st.session_state.is_admin = user_data['is_admin']
                            st.session_state.access_token = user_data['access_token']
                            st.session_state.logged_in = True
                            st.rerun()
                        else:
                            st.error("Failed to create account. Username or email might already exist.")
                    except Exception as e:
                        st.error(f"Registration error: {str(e)}")
                        logger.error(f"Registration error: {e}")
                else:
                    st.error("Please fill in all required fields and accept terms")

def recommender_page():
    st.title('🎬 Movie Recommender System')
    
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None

    # RNN Recommendations Section
    st.header("Watch Next: Recommended For You")
    st.write("Based on your recent rating history.")

    rnn_recommendations = api_client.get_rnn_recommendations(n=5)

    if rnn_recommendations:
        cols = st.columns(5)
        for i, movie in enumerate(rnn_recommendations):
            movie_id = int(movies[movies['title'] == movie['title']].iloc[0]['movie_id'])
            with cols[i % 5]:
                st.image(movie['poster'], use_container_width=True)
                st.caption(movie['title'])
                
                # Add to watchlist button
                if st.button("➕", key=f"rnn_watch_{movie_id}", help="Add to Watchlist"):
                    result = api_client.add_to_watchlist(movie_id, movie['title'])
                    if result:
                        st.toast(f"Added '{movie['title']}' to your watchlist!")
                    else:
                        st.toast(f"'{movie['title']}' is already in your watchlist.")
                
                with st.expander("More Info"):
                    st.write(f"**Release Date:** {movie['details']['release_date']}")
                    st.write(f"**Rating:** {movie['details']['vote_average']:.1f}/10")
                    if movie['details']['trailer_key']:
                        st.video(f"https://www.youtube.com/watch?v={movie['details']['trailer_key']}")
                    else:
                        st.text("No trailer available.")
    else:
        st.info("Rate a few more movies to get personalized 'Watch Next' recommendations!")

    st.markdown("---")

    # Genre Browser
    st.sidebar.header("Explore & Discover")
    all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 
                  'Fantasy', 'Horror', 'ScienceFiction', 'Thriller']
    display_genres = ["-"] + [g.replace('ScienceFiction', 'Science Fiction') for g in all_genres]
    selected_display_genre = st.sidebar.selectbox("Browse by Genre", display_genres)
    
    if selected_display_genre != "-":
        selected_genre = selected_display_genre.replace('Science Fiction', 'ScienceFiction')
        st.header(f"Top Movies in {selected_display_genre}")
        
        genre_movies_data = api_client.get_movies_by_genre(selected_genre, limit=10)
        
        if genre_movies_data:
            cols = st.columns(5)
            for i, movie_data in enumerate(genre_movies_data):
                with cols[i % 5]:
                    st.image(movie_data['poster'], use_container_width=True)
                    st.caption(movie_data['title'])
        else:
            st.write("No movies found for this genre in the current dataset.")
    
    st.sidebar.markdown("---")
    
    # Movie Selection for Recommendations
    st.sidebar.header("Get Recommendations")
    movie_list = movies['title'].values
    selected_title = st.sidebar.selectbox("Type or select a movie", options=movie_list, key='movie_selector')
    
    if st.sidebar.button('Get Recommendations'):
        st.session_state.selected_movie = selected_title

    if st.session_state.selected_movie:
        st.header(f"Recommendations for: *{st.session_state.selected_movie}*")
        
        recommended_movies = api_client.get_hybrid_recommendations(
            movie_title=st.session_state.selected_movie,
            n=5
        )
        
        if recommended_movies:
            cols = st.columns(5)
            for i, movie in enumerate(recommended_movies):
                movie_id = int(movies[movies['title'] == movie['title']].iloc[0]['movie_id'])
                with cols[i % 5]:
                    st.image(movie['poster'], use_container_width=True)
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button("➕", key=f"watch_{movie_id}", help="Add to Watchlist"):
                            result = api_client.add_to_watchlist(movie_id, movie['title'])
                            if result:
                                st.toast(f"Added '{movie['title']}' to your watchlist!")
                            else:
                                st.toast(f"'{movie['title']}' is already in your watchlist.")
                    
                    if st.button(movie['title'], key=f"title_{movie_id}"):
                        st.session_state.selected_movie = movie['title']
                        st.rerun()
                    
                    with st.expander("📋 More Info"):
                        st.write(f"**Release Date:** {movie['details']['release_date']}")
                        st.write(f"**Rating:** {movie['details']['vote_average']:.1f}/10")
                        st.markdown(f"**Overview:** {movie['details']['overview']}")
                        if movie['details']['trailer_key']:
                            st.video(f"https://www.youtube.com/watch?v={movie['details']['trailer_key']}")
                        else:
                            st.text("No trailer available.")
        else:
            st.error("Could not find recommendations for this movie.")
    else:
        st.info("Select a movie from the sidebar and click 'Get Recommendations' to start.")

def user_dashboard():
    st.title(f"Dashboard for {st.session_state.username}")
    tab1, tab2, tab3 = st.tabs(["My Watchlist", "My Ratings", "👤 Profile & Settings"])
    
    with tab1:
        st.header("🎬 Movies to Watch")
        try:
            watchlist_items = api_client.get_watchlist()
            
            if not watchlist_items:
                st.info("Your watchlist is empty.")
            else:
                cols = st.columns(4)
                for i, item in enumerate(watchlist_items):
                    with cols[i % 4]:
                        st.image(fetch_poster(item['movie_id']), use_container_width=True)
                        st.caption(item['movie_title'])
                        if st.button("🗑️ Remove", key=f"remove_watchlist_{item['id']}", 
                                   help="Remove from watchlist"):
                            if api_client.remove_from_watchlist(item['id']):
                                st.toast(f"Removed '{item['movie_title']}' from watchlist!")
                                st.rerun()
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            st.error("Could not load your watchlist.")
    
    with tab2:
        st.header("⭐ My Movie Ratings")
        try:
            user_ratings = api_client.get_ratings()
            
            if not user_ratings:
                st.info("You haven't rated any movies yet.")
            else:
                for rating in user_ratings:
                    movie_info = movies.loc[movies['movie_id'] == rating['movie_id']]
                    if not movie_info.empty:
                        movie_title = movie_info['title'].iloc[0]
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.image(fetch_poster(rating['movie_id']))
                            with col2:
                                st.subheader(movie_title)
                                new_rating_val = st.slider(
                                    "Update your rating", 
                                    1, 10, 
                                    int(rating['rating']), 
                                    key=f"dash_slider_{rating['id']}"
                                )
                                btn_col1, btn_col2, _ = st.columns([1, 1, 2])
                                with btn_col1:
                                    if st.button("Update", key=f"update_rating_{rating['id']}"):
                                        result = api_client.create_rating(
                                            rating['movie_id'], 
                                            float(new_rating_val)
                                        )
                                        if result:
                                            st.toast(f"Updated rating for '{movie_title}'!")
                                            st.rerun()
                                with btn_col2:
                                    if st.button("Delete", key=f"delete_rating_{rating['id']}"):
                                        if api_client.delete_rating(rating['id']):
                                            st.toast(f"Deleted your rating for '{movie_title}'!")
                                            st.rerun()
                            st.markdown("---")
        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            st.error("Could not load your ratings.")
    
    with tab3:
        st.header("Profile Information")
        try:
            profile = api_client.get_profile()
            if profile:
                st.text_input("Username", value=profile['username'], disabled=True)
                st.text_input("Email", value=profile['email'], disabled=True)
                created_at = pd.to_datetime(profile['created_at'])
                st.text_input("Member Since", 
                            value=created_at.strftime("%B %d, %Y"), 
                            disabled=True)
                
                with st.expander("Edit Profile"):
                    with st.form("profile_form"):
                        st.write("Leave a field blank to keep the current value.")
                        new_username = st.text_input("New Username", 
                                                    placeholder="Enter new username")
                        new_email = st.text_input("New Email", 
                                                 placeholder="Enter new email address")
                        submitted = st.form_submit_button("Save Changes")
                        
                        if submitted:
                            if not new_username and not new_email:
                                st.warning("Please enter a new username or email to update.")
                            else:
                                success, message = api_client.update_profile(
                                    new_username=new_username or None,
                                    new_email=new_email or None
                                )
                                if success:
                                    st.success(message)
                                    if new_username:
                                        st.session_state.username = new_username
                                    st.rerun()
                                else:
                                    st.error(message)
        except Exception as e:
            logger.error(f"Error loading profile: {e}")
            st.error("Could not load your profile.")

def feedback_page():
    st.title("📝 Submit Feedback")
    st.write("We value your feedback! Please let us know what you think.")
    
    with st.form("feedback_form"):
        feedback_text = st.text_area(
            "Your feedback", 
            height=150, 
            placeholder="Tell us about your experience, suggest a feature, or report a bug."
        )
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            if not feedback_text.strip():
                st.warning("Please enter some feedback before submitting.")
            else:
                success = api_client.submit_feedback(feedback_text)
                if success:
                    st.success("Thank you for your feedback! We appreciate you helping us improve.")
                else:
                    st.error("Sorry, we couldn't submit your feedback at this time. Please try again later.")

def search_and_rate_page():
    st.title("🔍 Search and Rate Movies")
    search_query = st.text_input("Enter a movie title to search", "")
    
    if search_query:
        results = api_client.search_movies(search_query, limit=20)
        
        if results:
            st.subheader(f"Found {len(results)} results for '{search_query}'")
            for movie_data in results:
                movie_id = movie_data['movie_id']
                movie_title = movie_data['title']
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(movie_data['poster'], use_container_width=True)
                with col2:
                    st.subheader(movie_title)
                    rating_val = st.slider(
                        "Your Rating (1-10)", 
                        1, 10, 5, 
                        key=f"search_rate_slider_{movie_id}"
                    )
                    if st.button("⭐ Rate", key=f"search_rate_btn_{movie_id}"):
                        result = api_client.create_rating(movie_id, float(rating_val))
                        if result:
                            st.toast(f"You rated '{movie_title}' {rating_val}/10!")
                        else:
                            st.error("Could not submit rating.")
                st.markdown("---")
        else:
            st.warning(f"No movies found matching '{search_query}'. Please try another title.")
    else:
        st.info("Type a movie title above to begin your search.")

def admin_dashboard_page():
    """Admin dashboard page using API"""
    st.title("📊 Admin Dashboard")
    st.write("Welcome to the admin panel. Here you can monitor application usage and performance.")

    try:
        stats_data = api_client.get_admin_stats()
        
        if not stats_data:
            st.error("Could not load admin statistics. Make sure you have admin privileges.")
            return
        
        metrics = stats_data.get('metrics', {})
        
        # Key Metrics Section
        st.header("Key Performance Indicators (KPIs)")
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Users", f"{metrics.get('total_users', 0)} 👤")
            col2.metric("Total Ratings Submitted", f"{metrics.get('total_ratings', 0)} ⭐")
            col3.metric("Total Feedback Entries", f"{metrics.get('total_feedback', 0)} 📝")
        else:
            st.warning("Could not load key metrics.")
        
        st.markdown("---")

        # Charts and Data Tables
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 Top 10 Most Rated Movies")
            most_rated = stats_data.get('most_rated_movies', [])
            if most_rated:
                most_rated_df = pd.DataFrame(most_rated)
                st.dataframe(most_rated_df, use_container_width=True)
            else:
                st.info("No rating data available yet.")

        with col2:
            st.subheader("📈 Top 10 Most Active Users")
            user_activity = stats_data.get('top_users', [])
            if user_activity:
                user_activity_df = pd.DataFrame(user_activity)
                st.bar_chart(user_activity_df.set_index('username'))
            else:
                st.info("No user activity available yet.")

        st.markdown("---")

        # User Feedback Section
        st.header("✉️ User Feedback")
        feedback_list = api_client.get_all_feedback()
        if feedback_list:
            for feedback in feedback_list:
                submitted_at = pd.to_datetime(feedback['submitted_at'])
                with st.expander(
                    f"Feedback from **{feedback['username']}** on "
                    f"*{submitted_at.strftime('%Y-%m-%d %H:%M')}*"
                ):
                    st.write(feedback['feedback_text'])
        else:
            st.info("No feedback has been submitted yet.")
            
    except Exception as e:
        logger.error(f"Error loading admin dashboard: {e}")
        st.error("Could not load admin dashboard.")

def main():
    # Set token if user is logged in
    if st.session_state.get('logged_in', False) and 'access_token' in st.session_state:
        api_client.set_token(st.session_state.access_token)
    
    if not st.session_state.get('logged_in', False):
        login_page()
        return

    st.sidebar.success(f"Logged in as {st.session_state.username}")
    
    # Navigation options
    nav_options = ["Movie Recommender", "Search & Rate", "My Dashboard", "Submit Feedback"]
    
    if st.session_state.get('is_admin', False):
        nav_options.append("Admin Dashboard")

    page = st.sidebar.radio("Navigation", nav_options)
    
    if st.sidebar.button("Logout"):
        for key in ['logged_in', 'user_id', 'username', 'is_admin', 'access_token']:
            if key in st.session_state:
                del st.session_state[key]
        api_client.set_token(None)
        st.rerun()

    # Page Routing
    if page == "Movie Recommender":
        recommender_page()
    elif page == "Search & Rate":
        search_and_rate_page()
    elif page == "My Dashboard":
        user_dashboard()
    elif page == "Submit Feedback":
        feedback_page()
    elif page == "Admin Dashboard":
        admin_dashboard_page()

if __name__ == '__main__':
    main()