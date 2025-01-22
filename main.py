import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Membaca dataset yang sudah diproses
df = pd.read_csv('games_dataset_cleaned.csv')

# Menampilkan beberapa data untuk memverifikasi
st.title("Sistem Rekomendasi Game")
st.dataframe(df.head())

# Persiapkan data untuk collaborative filtering
num_users = 100  # Anda bisa menyesuaikan jumlah pengguna
num_games = len(df)

# Generate data pengguna dan rating secara acak
ratings_data = {
    'user_id': np.random.choice(range(num_users), size=num_users * num_games),
    'game_id': np.tile(df.index, num_users),
    'rating': np.random.randint(1, 6, size=num_users * num_games)  # Rating acak antara 1 hingga 5
}

ratings_df = pd.DataFrame(ratings_data)

# Menampilkan data rating acak
st.write("Data Rating Acak:")
st.dataframe(ratings_df.head())

# Mempersiapkan dataset untuk surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'game_id', 'rating']], reader)

# Membagi data menjadi train dan test
trainset, testset = train_test_split(data, test_size=0.2)

# Membangun model menggunakan algoritma SVD
model = SVD()
model.fit(trainset)

# Fungsi rekomendasi berdasarkan prediksi SVD
def get_recommendations(user_id, top_n=5):
    user_ratings = [model.predict(user_id, game_id) for game_id in df.index]
    user_ratings.sort(key=lambda x: x.est, reverse=True)
    recommended_games = []
    for i, rating in enumerate(user_ratings[:top_n]):
        game_name = df.iloc[rating.iid]['Game Name']
        recommended_games.append(f"{i + 1}. {game_name}")
    return recommended_games

# Input nama pengguna dan memberikan rekomendasi
user_id_input = st.number_input("Masukkan User ID", min_value=0, max_value=num_users - 1, value=0)
top_n = st.slider("Jumlah Rekomendasi", min_value=1, max_value=10, value=5)

if st.button("Tampilkan Rekomendasi"):
    recommendations = get_recommendations(user_id_input, top_n)
    st.write("Top Recommeded Games for User:", user_id_input)
    for game in recommendations:
        st.write(game)
