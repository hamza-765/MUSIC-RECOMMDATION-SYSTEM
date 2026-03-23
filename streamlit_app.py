import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "model_outputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# ── Loaders ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_user_index():
    idx_path = os.path.join(PROCESSED_DIR, "user_index.parquet.pkl")
    par_path = os.path.join(PROCESSED_DIR, "features_deploy.parquet")

    if os.path.exists(idx_path):
        with open(idx_path, "rb") as f:
            return pickle.load(f)

    if os.path.exists(par_path):
        df = pd.read_parquet(par_path)
        return {
            user: group.reset_index(drop=True)
            for user, group in df.groupby("user")
        }

    raise FileNotFoundError(
        f"No feature file found. Files in processed/: "
        f"{os.listdir(PROCESSED_DIR) if os.path.exists(PROCESSED_DIR) else 'folder missing'}"
    )

# ── Load everything ───────────────────────────────────────────
try:
    model, feat_cols = load_model()
    user_index       = load_user_index()
    user_list        = sorted(user_index.keys())

    # Build summary stats from index directly — no features_df needed
    total_users = len(user_list)
    total_songs = sum(len(v) for v in user_index.values())
    sample_df   = next(iter(user_index.values()))
    repeat_rate = sum(
        v["label"].mean() for v in user_index.values()
    ) / total_users

except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Repeat Predictor",
    page_icon="🎵",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────
st.title("🎵 Spotify Repeat Play Predictor")
st.markdown("Predict whether a user will replay a song within **30 days**.")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.success("Model loaded ✅")
st.sidebar.markdown(f"**Features:** {len(feat_cols)}")
st.sidebar.markdown(f"**Users:** {total_users:,}")
st.sidebar.markdown(f"**Songs:** {total_songs:,}")
st.sidebar.markdown(f"**Repeat rate:** {repeat_rate:.1%}")
st.sidebar.divider()
st.sidebar.header("⚙️ Settings")
top_n     = st.sidebar.slider("Top N Recommendations", 5, 20, 10)
threshold = st.sidebar.slider("Repeat Threshold", 0.1, 0.9, 0.5, step=0.05)

st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📋 Recommendations", "📊 Model Info"])


# ── TAB 1 · Predict ───────────────────────────────────────────
with tab1:
    st.subheader("Single Song Prediction")

    sample_ids = user_list[:3]
    st.info(f"Sample user IDs: `{'`, `'.join(sample_ids)}`")

    col1, col2 = st.columns(2)
    with col1:
        user_id   = st.text_input("User ID", value=sample_ids[0], key="predict_user")

    with col2:
        user_rows = user_index.get(user_id, pd.DataFrame())
        if not user_rows.empty:
            song_id = st.selectbox("Song ID", options=user_rows["song"].tolist(), key="predict_song")
        else:
            song_id = st.text_input("Song ID", key="predict_song_text")
            if user_id:
                st.warning(f"User '{user_id}' not found.")

    if st.button("🔮 Predict", type="primary", key="btn_predict"):
        if user_rows.empty:
            st.warning("User not found in dataset.")
        else:
            row = user_rows[user_rows["song"] == song_id]
            if row.empty:
                st.warning("Song not found for this user.")
            else:
                prob   = float(model.predict(row[feat_cols].fillna(-1))[0])
                actual = int(row["label"].values[0])

                st.divider()
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Repeat Probability", f"{prob:.1%}")
                col_b.metric("Prediction", "✅ Will Repeat" if prob >= threshold else "❌ Won't Repeat")
                col_c.metric("Actual Label", "Repeat ✅" if actual == 1 else "No Repeat ❌")
                st.progress(prob)

                if int(prob >= threshold) == actual:
                    st.success("Model got it **correct**!")
                else:
                    st.error(
                        f"Model got it **wrong**. "
                        f"Predicted {'repeat' if prob >= threshold else 'no repeat'}, "
                        f"actual was {'repeat' if actual == 1 else 'no repeat'}."
                    )


# ── TAB 2 · Recommendations ───────────────────────────────────
with tab2:
    st.subheader("Top Song Recommendations")

    rec_user = st.selectbox(
        "Select a User",
        options=user_list[:200],
        key="rec_user",
    )

    if st.button("📋 Get Recommendations", type="primary", key="btn_recommend"):
        user_rows = user_index.get(rec_user, pd.DataFrame())

        if user_rows.empty:
            st.warning("User not found.")
        else:
            with st.spinner("Scoring songs..."):
                user_rows = user_rows.copy()
                user_rows["repeat_prob"] = model.predict(
                    user_rows[feat_cols].fillna(-1)
                )
                top = (
                    user_rows
                    .sort_values("repeat_prob", ascending=False)
                    .head(top_n)[["song", "repeat_prob", "label"]]
                    .reset_index(drop=True)
                )
                top.index += 1

            st.success(f"Top {top_n} songs for **{rec_user}**")
            st.dataframe(top, use_container_width=True)

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
    col1.metric("Total Users",  f"{total_users:,}")
    col2.metric("Total Songs",  f"{total_songs:,}")
    col3.metric("Repeat Rate",  f"{repeat_rate:.1%}")
    col4.metric("Features",     f"{len(feat_cols)}")
