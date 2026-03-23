import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "model_outputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
        model, feat_cols = pickle.load(f)
    return model, feat_cols

@st.cache_resource
def load_user_index():
    # Try pre-built index first (fastest)
    idx_path = os.path.join(PROCESSED_DIR, "user_index.parquet.pkl")
    if os.path.exists(idx_path):
        with open(idx_path, "rb") as f:
            return pickle.load(f)

    # Fallback — build from parquet
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "features_deploy.parquet"))
    return {
        user: group.reset_index(drop=True)
        for user, group in df.groupby("user")
    }

model, feat_cols = load_model()
user_index       = load_user_index()
user_list        = sorted(user_index.keys())

st.set_page_config(
    page_title="Spotify Repeat Predictor",
    page_icon="🎵",
    layout="wide",
)

# ── Load model directly (no API needed) ──────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
        model, feat_cols = pickle.load(f)
    return model, feat_cols

@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))

try:
    model, feat_cols = load_model()
    features_df      = load_features()
    model_loaded     = True
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ── Header ────────────────────────────────────────────────────
st.title("🎵 Spotify Repeat Play Predictor")
st.markdown("Predict whether a user will replay a song within **30 days**.")
st.sidebar.success("Model loaded ✅")
st.sidebar.markdown(f"**Features:** {len(feat_cols)}")
st.sidebar.markdown(f"**Users:** {features_df['user'].nunique():,}")
st.sidebar.markdown(f"**Songs:** {features_df['song'].nunique():,}")
st.divider()

# ── Sidebar settings ──────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
top_n     = st.sidebar.slider("Top N Recommendations", 5, 20, 10)
threshold = st.sidebar.slider("Repeat Threshold", 0.1, 0.9, 0.5, step=0.05)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📋 Recommendations", "📊 Model Info"])


# ── TAB 1 · Predict ───────────────────────────────────────────
with tab1:
    st.subheader("Single Song Prediction")

    # Show sample user IDs
    sample_ids = features_df["user"].unique()[:5].tolist()
    st.info(f"Sample user IDs: `{'`, `'.join(sample_ids[:3])}`")

    col1, col2 = st.columns(2)
    with col1:
        user_id = st.text_input("User ID", value=sample_ids[0], key="predict_user")
    with col2:
        # Show songs for selected user
        user_songs = features_df[features_df["user"] == user_id]["song"].tolist()
        if user_songs:
            song_id = st.selectbox("Song ID", options=user_songs, key="predict_song")
        else:
            song_id = st.text_input("Song ID", key="predict_song_text")

    if st.button("🔮 Predict", type="primary", key="btn_predict"):
        row = features_df[
            (features_df["user"] == user_id) &
            (features_df["song"] == song_id)
        ]

        if row.empty:
            st.warning("User/Song combination not found in dataset.")
        else:
            X    = row[feat_cols].fillna(-1)
            prob = float(model.predict(X)[0])
            actual = int(row["label"].values[0])

            st.divider()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Repeat Probability", f"{prob:.1%}")
            col_b.metric("Prediction", "✅ Will Repeat" if prob >= threshold else "❌ Won't Repeat")
            col_c.metric("Actual Label", "Repeat ✅" if actual == 1 else "No Repeat ❌")

            st.progress(prob)

            if int(prob >= threshold) == actual:
                st.success(f"Model got it **correct**! Predicted {'repeat' if prob >= threshold else 'no repeat'}, actual was {'repeat' if actual == 1 else 'no repeat'}.")
            else:
                st.error(f"Model got it **wrong**. Predicted {'repeat' if prob >= threshold else 'no repeat'}, actual was {'repeat' if actual == 1 else 'no repeat'}.")


# ── TAB 2 · Recommendations ───────────────────────────────────
with tab2:
    st.subheader("Top Song Recommendations")

    sample_ids = features_df["user"].unique()[:10].tolist()
    st.info(f"Sample user IDs: `{'`, `'.join(sample_ids[:3])}`")

    rec_user = st.selectbox(
        "Select a User",
        options=features_df["user"].unique()[:100],
        key="rec_user"
    )

    if st.button("📋 Get Recommendations", type="primary", key="btn_recommend"):
        user_rows = features_df[features_df["user"] == rec_user].copy()

        if user_rows.empty:
            st.warning("User not found.")
        else:
            user_rows["repeat_prob"] = model.predict(user_rows[feat_cols].fillna(-1))
            top = (
                user_rows
                .sort_values("repeat_prob", ascending=False)
                .head(top_n)[["song", "repeat_prob", "label"]]
                .reset_index(drop=True)
            )
            top.index += 1
            top["correct"] = (
                ((top["repeat_prob"] >= threshold) & (top["label"] == 1)) |
                ((top["repeat_prob"] <  threshold) & (top["label"] == 0))
            )

            st.success(f"Top {top_n} songs for **{rec_user}**")
            st.dataframe(top, use_container_width=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#5563D4" if p >= threshold else "#B0AEC9"
                      for p in top["repeat_prob"]]
            ax.barh(top["song"], top["repeat_prob"], color=colors)
            ax.axvline(threshold, color="#E8593C", linestyle="--",
                       label=f"Threshold ({threshold})")
            ax.set_xlabel("Repeat Probability")
            ax.set_title(f"Top {top_n} Songs — {rec_user}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Songs Analysed",    len(user_rows))
            col2.metric("Predicted Repeats", int((top["repeat_prob"] >= threshold).sum()))
            col3.metric("Actual Repeats",    int(top["label"].sum()))


# ── TAB 3 · Model Info ────────────────────────────────────────
with tab3:
    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pipeline Steps**")
        steps = [
            ("1", "Load Data",           "Music4All CSVs"),
            ("2", "Preprocess",          "Parse timestamps, filter"),
            ("3", "Target Variable",     "Repeat within 30 days"),
            ("4", "Collaborative Feats", "User & song stats"),
            ("5", "Temporal Feats",      "Hour, day, recency"),
            ("6", "Content Feats",       "Audio, genres, language"),
            ("7", "SVD Embeddings",      "64-dim latent vectors"),
            ("8", "LightGBM",            "Binary classifier"),
        ]
        for num, name, desc in steps:
            st.markdown(f"**{num}.** {name} — `{desc}`")

    with col2:
        st.markdown("**Model Parameters**")
        params = {
            "Algorithm":      "LightGBM",
            "Objective":      "Binary classification",
            "Metric":         "AUC-ROC",
            "Embeddings":     "Truncated SVD (64 factors)",
            "Repeat window":  "30 days",
            "Learning rate":  "0.05",
            "Num leaves":     "64",
            "Early stopping": "30 rounds",
        }
        st.table(pd.DataFrame(params.items(), columns=["Parameter", "Value"]))

    st.divider()
    st.markdown("**Dataset Stats**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", f"{len(features_df):,}")
    col2.metric("Users",         f"{features_df['user'].nunique():,}")
    col3.metric("Songs",         f"{features_df['song'].nunique():,}")
    col4.metric("Repeat Rate",   f"{features_df['label'].mean():.1%}")
