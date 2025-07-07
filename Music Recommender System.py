import streamlit as st
import pandas as pd
import numpy as np
import os

BASE_DIR = os.getcwd()
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
DATA_DIR = os.path.join(RESOURCES_DIR, 'data')
MODEL_DIR = os.path.join(RESOURCES_DIR, 'models')
MATRICES_DIR = os.path.join(RESOURCES_DIR, 'matrices')


@st.cache_data
def load_csv(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"File {file_name} not found in {base_path}")
        return None
    

df_gym = load_csv(DATA_DIR, 'modified_gym_members_exercise_tracking.csv')

if df_gym is None:
    st.error("Error loading data files. Please check the files in the resources/data directory.")
    st.stop()


st.markdown("# Music Recommender System")

df_users_shown = df_gym[['Age', 'Gender', 'Weight (kg)','Height (m)', 'Session_Duration (hours)', 'Workout_Type']].copy()
df_users_shown.rename(columns={'Session_Duration (hours)': 'Duration (hours)'}, inplace=True) 
df_users_shown.index.name = "ID"
df_users_shown.index = range(1, len(df_users_shown) + 1)
st.dataframe(df_users_shown, height=200)

