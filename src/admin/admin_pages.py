import streamlit as st
import pandas as pd
from .admin_manager import AdminManager
from src.recommender import movies

def admin_dashboard_page():
    """
    It is the main page for the admin dashboard.
    """
    st.title("📊 Admin Dashboard")
    st.write("Welcome to the admin panel. Here you can monitor application usage and performance.")

    admin_manager = AdminManager()

    # Key Metrics Section 
    st.header("Key Performance Indicators (KPIs)")
    metrics = admin_manager.get_key_metrics()
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", f"{metrics['total_users']} 👤")
        col2.metric("Total Ratings Submitted", f"{metrics['total_ratings']} ⭐")
        col3.metric("Total Feedback Entries", f"{metrics['total_feedback']} 📝")
    else:
        st.warning("Could not load key metrics.")
    
    st.markdown("---")

    # Charts and Data Tables
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎬 Top 10 Most Rated Movies")
        most_rated_df = admin_manager.get_most_rated_movies(movies, limit=10)
        if not most_rated_df.empty:
            st.dataframe(most_rated_df, use_container_width=True)
        else:
            st.info("No rating data available yet.")

    with col2:
        st.subheader("📈 Top 10 Most Active Users")
        user_activity_df = admin_manager.get_user_activity(limit=10)
        if not user_activity_df.empty:
            st.bar_chart(user_activity_df.set_index('username'))
        else:
            st.info("No user activity available yet.")

    st.markdown("---")

    # User Feedback Section
    st.header("✉️ User Feedback")
    feedback_list = admin_manager.get_all_feedback()
    if feedback_list:
        for _, username, submitted_at, feedback_text in feedback_list:
            with st.expander(f"Feedback from **{username}** on *{submitted_at.strftime('%Y-%m-%d %H:%M')}*"):
                st.write(feedback_text)
    else:
        st.info("No feedback has been submitted yet.")
