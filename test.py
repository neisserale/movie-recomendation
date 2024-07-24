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
    print(type(movies))
    #print(df_recommend[~df_recommend['id'].isin(movies)])
    ## eliminación de las movies vistas
    df_recommend = df_recommend[~df_recommend['id'].isin(movies)]
    df_recommend = df_recommend.sort_values('distance', ascending=True).reset_index(drop=True)
    df_recommend = df_recommend.head(k)

    return df_recommend.to_json()

abc = recommend_movies_by_user(94375, knn_model, tfidf_movies_df)

for movie in [abc]:
    print(movie)

