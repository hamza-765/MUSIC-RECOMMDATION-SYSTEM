# 🎵 Spotify Repeat Play Predictor

Predicts whether a user will replay a song within 30 days using the Music4All dataset.

## 🔗 Live Demo
[Click here to try the app](https://music-recommdation-system-2025tarnished.streamlit.app/)

## 🏗️ Tech Stack
- **ML Model:** LightGBM + SVD Embeddings
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Dataset:** Music4All (14,125 users, 80,735 songs)

## 📊 Pipeline
1. Data preprocessing & filtering
2. Collaborative + temporal + content features
3. SVD matrix factorisation (64 dimensions)
4. LightGBM binary classifier
5. Streamlit web app

## 🚀 Run Locally
git clone https://github.com/hamza-765/MUSIC-RECOMMDATION-SYSTEM
cd MUSIC-RECOMMDATION-SYSTEM
pip install -r requirements.txt
streamlit run streamlit_app.py

## 📁 Project Structure
app/                  FastAPI backend
model_outputs/        Trained model artifacts
pipeline/             Training pipeline
processed/            Feature files
streamlit_app.py      Streamlit frontend
