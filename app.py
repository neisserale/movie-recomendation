import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO

import pickle
import json


## read files
df_unknown = pd.read_csv('data/ratings.test.unknown.csv')
tfidf_movies_df = pd.read_csv('data/tfidf_movies_df.csv')
profiling = pd.read_csv('data/profiling.csv')

with open('data/feature_names.json', 'r') as f:
    feature_names = json.load(f)

with open('data/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

## list of user
users = list(df_unknown['userId'].unique())

## list of popular movies
popular_movies = [
    {"title": "The Million Dollar Hotel", 'ratio':4.44, "poster_url": "img/The Million Dollar Hotel.jpg"},
    {"title": "Terminator 3: Rise of the Machines", 'ratio':4.14, "poster_url": "img/Terminator 3 Rise of the Machines.jpg"},
    {"title": "Solaris", 'ratio':4.12, "poster_url": "img/Solaris.jpg"},
    {"title": "The 39 Steps", 'ratio':4.15, "poster_url": "img/The 39 Steps.jpg"},
    {"title": "Monsoon Wedding", 'ratio':3.7, "poster_url": "img/Monsoon Wedding.jpg"},
]

## recommender system
def recommend_movies_by_user(user, model, df_movies, k=10, feature_names=feature_names):
    ## obtención del profiling del user
    values = profiling[profiling['userId'] == user].iloc[:, 2:]

    ## obtención de la distancia de las movies
    distances, indices = model.kneighbors(values, n_neighbors=len(df_movies))
    recommended_movies = df_movies.iloc[indices.flatten()]['title']

    df_recommend = pd.DataFrame({'index': indices.flatten(),
                                 'id': df_movies.iloc[indices.flatten()]['id'],
                                 'title': df_movies.iloc[indices.flatten()]['title'],
                                 'distance': distances.flatten()})

    ## obtención de las movies ya vistas por el user
    movies = eval(profiling[profiling['userId'] == user]['id'].values[0])

    ## eliminación de las movies vistas
    df_recommend = df_recommend[~df_recommend['id'].isin(movies)]
    df_recommend = df_recommend.sort_values('distance', ascending=True).reset_index(drop=True)
    df_recommend = df_recommend.head(k)

    return df_recommend

## set main page
st.set_page_config(layout="wide")

## title of the app
st.title("Sistema de Recomendación de Películas - Amazon 3000")

##################################################################
## System Recomendation
##################################################################
## user selection
def format_func(option):
    if option == "":
        return "Selecciona un usuario"  # placeholder text
    else:
        return option

st.subheader("Recomendaciones Personalizadas")

## adjust the width by changing the number
col = st.columns(4)
with col[0]:
    selected_user = st.selectbox(label='Lista de Usuarios', options=[''] + users, index=0, format_func=format_func)
   

## show recommendations
if selected_user != "":
    st.subheader(f"Películas recomendadas para el usuario: {selected_user}")

    ## get the recomendation
    recommended_movies = recommend_movies_by_user(selected_user, knn_model, tfidf_movies_df)

    cols = st.columns(5)
    list_movies = recommended_movies['title'].unique()      
    for idx, movie in enumerate(list_movies):
        with cols[idx % 5]:
            st.text(movie)

##################################################################
## Popularity Movies
##################################################################

st.header("Películas más vistas por la comunidad!")

cols = st.columns(6)
for idx, movie in enumerate(popular_movies):
    with cols[idx % 5]:
        st.text(movie["title"])
        rating = movie["ratio"]
        stars = ''.join(['★' for _ in range(int(rating))]) + ''.join(['☆' for _ in range(5 - int(rating))])
        st.text(f'{stars} ({rating}/5)')

        if os.path.exists(movie["poster_url"]):
            img = Image.open(movie["poster_url"])
            st.image(img)
        else:
            st.write("Image not found")

print('Here!!!')