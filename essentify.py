import streamlit as st
import os.path
import json
import pandas as pd
from datetime import datetime

from utils.collection import Collection

collection = Collection()
with open('models/metadata/genre_discogs400-discogs-effnet-1.json', 'r') as f:
    mappings = json.load(f)['classes']

def save_playlist(filenames):
    # Playlist name and path
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    extension = '.m3u'
    name = timestamp + extension
    dirname = 'playlists'
    os.makedirs(dirname, exist_ok=True)
    path = os.path.join(dirname, name)

    # Write to m3u file
    with open(path, "w", encoding="utf-8") as f:
        for filename in filenames:
            f.write(f"{os.path.join('..', filename)}\n")


def search_similar_tracks(filename):
    # Filter by similarity
    results = collection.search_similar_tracks(filename, embedding_model.lower())

    # Get top 10 tracks
    results = results.head(num_tracks)

    # Get filenames
    filenames = results.index

    # Save button
    save = st.button('Save playlist', on_click=lambda f=filenames: save_playlist(f))
    

    # Display results
    for filename in filenames:
        with st.container():
            st.write(filename.split('/')[-1])
            
            col1, col2 = st.columns(2)

            with col1:
                st.audio(filename, format="audio/mp3")

            with col2:
                st.button('Search similar tracks', on_click=lambda f=filename: search_similar_tracks(f), key=filename)


def filter_results():
    # Filter by style
    results = collection.sort_by_style(mappings.index(style.replace('-', '---')))

    # Filter by tempo
    results = collection.filter_by_tempo(results, tempo)

    # Filter by instrumentals
    results = collection.filter_instrumentals(results, require_instrumentals)

    # Filter by dancability
    results = collection.filter_by_dancability(results, dancability)
    
    # Filter by arousal and valence
    results = collection.filter_by_arousal_and_valence(results, arousal, valence)

    # Filter by key and scale
    results = collection.filter_by_key_and_scale(results, key, scale)

    # Get top 10 tracks
    results = results.head(num_tracks)
    
    # Get filenames
    filenames = results.index

    # Save button
    save = st.button('Save playlist', on_click=lambda f=filenames: save_playlist(f))
    
    # Display results
    for filename in filenames:
        with st.container():
            st.write(filename.split('/')[-1])
            
            col1, col2 = st.columns(2)

            with col1:
                st.audio(filename, format="audio/mp3")

            with col2:
                st.button('Search similar tracks', on_click=lambda f=filename: search_similar_tracks(f), key=filename)


with st.sidebar:
    # Music style dropdown menu
    style = st.selectbox('Style', options=[c.replace('---', '-') for c in mappings], index=0)

    # Tempo slider
    tempo = st.slider('Tempo', min_value=60, max_value=240, value=120)
    
    # Voice/Instrumental toggle button
    require_instrumentals = st.toggle('Instrumental', value=False)

    # Dancability range selector
    dancability = st.select_slider('Dancability', options=range(100 + 1), value=(0, 100))

    # Arousal range selector
    arousal = st.select_slider('Arousal', options=range(10), value=(0, 9))

    # Valence range selector
    valence = st.select_slider('Valence', options=range(10), value=(0, 9))
    
    col1, col2 = st.columns(2)
    
    # Key dropdown menu
    with col1:
        key = st.selectbox('Key', options=['All', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'], index=0)

    # Scale dropdown menu
    with col2:
        scale = st.selectbox('Scale', options=['All', 'Major', 'Minor'], index=0)

    # Embedding model dropdown menu
    embedding_model = st.selectbox('Embeddings', options=['Effnet', 'MusiCNN'], index=0)

    # Number of tracks input
    num_tracks = st.number_input('Number of tracks', min_value=1, max_value=50, value=10, step=1)
    
    # Search button
    button = st.button('Search', on_click=filter_results)

