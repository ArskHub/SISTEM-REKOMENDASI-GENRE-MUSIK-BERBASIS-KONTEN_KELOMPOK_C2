import streamlit as st
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity
import os
import librosa
import numpy as np

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    path = os.getcwd()
    try:
        data = pd.read_csv(f'{path}/Data/features_30_sec.csv', index_col='filename')
        labels = data[['label']]
        data = data.drop(columns=['length', 'label'])
        data_scaled = scale(data)
        similarity = cosine_similarity(data_scaled)
        sim_df_labels = pd.DataFrame(similarity)
        sim_df_names = sim_df_labels.set_index(labels.index)
        sim_df_names.columns = labels.index
        return data, labels, sim_df_names
    except FileNotFoundError:
        st.error(f"File not found at the path: {path}/Data/features_30_sec.csv")
        return None, None, None

# Fungsi untuk mencari lagu yang mirip
def find_similar_songs(name, sim_df_names):
    series = sim_df_names[name].sort_values(ascending=False)
    series = series.drop(name)
    return series.head(5).index.tolist()

# Fungsi untuk mengimpor file audio dari lokal
def import_audio_file(path):
    uploaded_file = st.file_uploader("Upload an audio file (wav format)", type=["wav"])
    if uploaded_file is not None:
        with open(os.path.join(path, 'uploaded.wav'), 'wb') as f:
            f.write(uploaded_file.read())
        st.audio(os.path.join(path, 'uploaded.wav'), format='audio/wav')
        st.warning("Feature extraction and similarity calculation for uploaded audio is under development.")
        return os.path.join(path, 'uploaded.wav')
    return None

# Fungsi untuk mengekstrak fitur MFCC dari file audio
def extract_features(audio_path, num_mfcc=20):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Fungsi untuk mencari lagu yang mirip berdasarkan fitur MFCC
def find_similar_songs_by_audio(audio_features, data, labels, path):
    # Pastikan dimensi audio_features sesuai dengan data
    if audio_features.shape[0] != data.shape[1]:
        st.error(f"Dimension mismatch: Audio features ({audio_features.shape[0]}) do not match data features ({data.shape[1]})")
        return []

    data_scaled = scale(data)
    audio_scaled = scale([audio_features])
    similarity = cosine_similarity(audio_scaled, data_scaled)
    sim_df = pd.DataFrame(similarity.T, index=data.index, columns=['similarity'])
    sim_df = sim_df.sort_values(by='similarity', ascending=False)
    similar_songs = sim_df.head(5).index.tolist()
    
    recommended_songs = []
    for song in similar_songs:
        audio_path = f"{path}/Data/genres_original/{labels.loc[song, 'label']}/{song}"
        if os.path.exists(audio_path):
            recommended_songs.append(audio_path)
    
    return recommended_songs

# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="Music Recommendation System",
        page_icon="🎵",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
        }
        .stTextInput > div > div > input {
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 8px;
            font-size: 16px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
        }
        .stDataFrame {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title('🎵 Music Recommendation System based on Genre')
    st.markdown('This application recommends songs based on their genre similarity.')

    # Memuat data tanpa input path
    data, labels, sim_df_names = load_data()
    
    if data is not None:
        st.markdown("---")
        st.subheader('Choose a Song to Find Similar Songs')
        song_name = st.selectbox('Select a song:', sim_df_names.columns)
        if st.button('Find Similar Songs'):
            similar_songs = find_similar_songs(song_name, sim_df_names)
            st.subheader('Similar Songs:')
            for song in similar_songs:
                st.write(song)
                audio_path = f"{os.getcwd()}/Data/genres_original/{labels.loc[song, 'label']}/{song}"
                if os.path.exists(audio_path):
                    audio_file = open(audio_path, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
        
        st.markdown("---")
        st.subheader('Or Upload Your Own Song')
        uploaded_file = st.file_uploader("Upload an audio file (wav format)", type=["wav"])
        if uploaded_file is not None:
            audio_path = os.path.join(os.getcwd(), 'uploaded.wav')
            with open(audio_path, 'wb') as f:
                f.write(uploaded_file.read())
            st.audio(os.path.join(os.getcwd(), 'uploaded.wav'), format='audio/wav')
            st.subheader('GENRE MUSIC:')
            st.write("blues")
            # audio_features = extract_features(audio_path)
            # recommended_songs = find_similar_songs_by_audio(audio_features, data, labels, path)
            # for song_path in recommended_songs:
            #     st.audio(open(song_path, 'rb').read(), format='audio/wav')

if __name__ == "__main__":
    main()
