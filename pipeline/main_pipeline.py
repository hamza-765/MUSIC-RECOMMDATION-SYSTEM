# ============================================================
#  Spotify Repeat Play Prediction — ML Pipeline
#  Dataset : Music4All
#  Model   : SVD Embeddings + LightGBM Classifier
# ============================================================

# ── STEP 1 · Install & Import Libraries ─────────────────────
# !pip install lightgbm scikit-learn pandas numpy scipy matplotlib python-dotenv --quiet

import os
import pickle
import sys
import warnings
from typing import Optional, Tuple, List

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Use non-interactive backend when running on a server (no display)
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

warnings.filterwarnings("ignore")
load_dotenv()  # loads variables from a .env file if present


# ── STEP 2 · Configuration ───────────────────────────────────

# Set DATA_DIR in a .env file or as an environment variable.
# Falls back to ./data/music4all for local development.
DATA_DIR            = os.getenv("DATA_DIR", "./data/music4all")
REPEAT_WINDOW_DAYS  = int(os.getenv("REPEAT_WINDOW_DAYS", 30))
MIN_PLAYS_PER_USER  = int(os.getenv("MIN_PLAYS_PER_USER", 5))
MIN_PLAYS_PER_SONG  = int(os.getenv("MIN_PLAYS_PER_SONG", 5))
SVD_FACTORS         = int(os.getenv("SVD_FACTORS", 32))
PROCESSED_DIR       = os.getenv("PROCESSED_DIR", "./processed")
MODEL_DIR           = os.getenv("MODEL_DIR", "./model_outputs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,     exist_ok=True)

NON_FEATURE_COLS = [
    "user", "song", "first_play", "label",
    "user_idx", "song_idx",
]

LEAKY_COLS = [
    "total_plays",
    "us_play_count",
    "us_days_span",
    "user_avg_plays_per_song",
    "song_avg_plays_per_listener",
]

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "learning_rate":     0.05,
    "num_leaves":        64,
    "min_child_samples": 30,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "verbose":           -1,
}


# ── STEP 3 · Load Data ───────────────────────────────────────

def load_csv(filename: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_DIR, filename)
    try:
        df = pd.read_csv(path, sep="\t")
        print(f"  ✅ {filename:<35} → {len(df):>8,} rows, {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        print(f"  ⚠️  {filename} not found — skipping")
        return None


def load_all_data() -> dict:
    print("Loading CSV files...")
    return {
        "history":  load_csv("listening_history.csv"),
        "genres":   load_csv("id_genres.csv"),
        "tags":     load_csv("id_tags.csv"),
        "metadata": load_csv("id_metadata.csv"),
        "lang":     load_csv("id_lang.csv"),
    }


# ── STEP 4 · Preprocess Listening History ───────────────────

def preprocess_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    except Exception:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, infer_datetime_format=True
        )

    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    user_counts  = df["user"].value_counts()
    song_counts  = df["song"].value_counts()
    active_users = user_counts[user_counts >= MIN_PLAYS_PER_USER].index
    active_songs = song_counts[song_counts >= MIN_PLAYS_PER_SONG].index

    df = df[
        df["user"].isin(active_users) & df["song"].isin(active_songs)
    ].reset_index(drop=True)

    print(f"  Events: {len(df):,} | Users: {df['user'].nunique():,} | Songs: {df['song'].nunique():,}")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


# ── STEP 5 · Build Target Variable ──────────────────────────

def build_target(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    print("Building labels...")
    df = df.sort_values(["user", "song", "timestamp"]).reset_index(drop=True)

    grp = df.groupby(["user", "song"])
    agg = grp["timestamp"].agg(
        first_play="min",
        last_play="max",
        total_plays="count",
    ).reset_index()

    df["rank"] = grp["timestamp"].rank(method="first")
    second_plays = (
        df[df["rank"] == 2][["user", "song", "timestamp"]]
        .rename(columns={"timestamp": "second_play"})
    )

    agg = agg.merge(second_plays, on=["user", "song"], how="left")
    window = pd.Timedelta(days=window_days)
    agg["label"] = (
        agg["second_play"].notna()
        & (agg["second_play"] - agg["first_play"] <= window)
    ).astype(int)

    target_df = agg[["user", "song", "first_play", "total_plays", "label"]].copy()
    pos = target_df["label"].sum()
    print(f"  Pairs: {len(target_df):,} | Repeat=1: {pos:,} ({100*pos/len(target_df):.1f}%)")
    return target_df


# ── STEP 6 · Collaborative Features ─────────────────────────

def build_collaborative_features(
    history: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    user_feats = history.groupby("user").agg(
        user_total_plays=("song", "count"),
        user_unique_songs=("song", "nunique"),
    ).reset_index()
    user_feats["user_avg_plays_per_song"] = (
        user_feats["user_total_plays"] / user_feats["user_unique_songs"]
    )

    song_feats = history.groupby("song").agg(
        song_total_plays=("user", "count"),
        song_unique_listeners=("user", "nunique"),
    ).reset_index()
    song_feats["song_avg_plays_per_listener"] = (
        song_feats["song_total_plays"] / song_feats["song_unique_listeners"]
    )

    us_feats = history.groupby(["user", "song"]).agg(
        us_play_count=("timestamp", "count"),
        us_first_play=("timestamp", "min"),
        us_last_play=("timestamp", "max"),
    ).reset_index()
    us_feats["us_days_span"] = (
        us_feats["us_last_play"] - us_feats["us_first_play"]
    ).dt.total_seconds() / 86400

    df = target_df.copy()
    df = df.merge(user_feats, on="user", how="left")
    df = df.merge(song_feats, on="song", how="left")
    df = df.merge(
        us_feats[["user", "song", "us_play_count", "us_days_span"]],
        on=["user", "song"], how="left",
    )
    print(f"  Collaborative features added → shape {df.shape}")
    return df


# ── STEP 7 · Temporal Features ──────────────────────────────

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["first_play_hour"]       = df["first_play"].dt.hour
    df["first_play_dow"]        = df["first_play"].dt.dayofweek
    df["first_play_month"]      = df["first_play"].dt.month
    df["is_weekend"]            = (df["first_play_dow"] >= 5).astype(int)
    latest = df["first_play"].max()
    df["days_since_first_play"] = (
        latest - df["first_play"]
    ).dt.total_seconds() / 86400
    print(f"  Temporal features added → shape {df.shape}")
    return df


# ── STEP 8 · Content Features ───────────────────────────────

def build_content_features(
    df: pd.DataFrame,
    genres_df:   Optional[pd.DataFrame],
    tags_df:     Optional[pd.DataFrame],
    metadata_df: Optional[pd.DataFrame],
    lang_df:     Optional[pd.DataFrame],
) -> pd.DataFrame:
    df = df.copy()

    if genres_df is not None:
        g = genres_df.copy()
        g.columns = [c.strip().lower() for c in g.columns]
        id_col, genre_col = g.columns[0], g.columns[1]
        g["genre_count"] = g[genre_col].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )
        df = df.merge(
            g[[id_col, "genre_count"]].rename(columns={id_col: "song"}),
            on="song", how="left",
        )

    if tags_df is not None:
        t = tags_df.copy()
        t.columns = [c.strip().lower() for c in t.columns]
        id_col, tag_col = t.columns[0], t.columns[1]
        t["tag_count"] = t[tag_col].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )
        df = df.merge(
            t[[id_col, "tag_count"]].rename(columns={id_col: "song"}),
            on="song", how="left",
        )

    if metadata_df is not None:
        m = metadata_df.copy()
        m.columns = [c.strip().lower() for c in m.columns]
        id_col = m.columns[0]
        audio_features = [
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "duration_ms",
        ]
        available = [c for c in audio_features if c in m.columns]
        if available:
            df = df.merge(
                m[[id_col] + available].rename(columns={id_col: "song"}),
                on="song", how="left",
            )

    if lang_df is not None:
        l = lang_df.copy()
        l.columns = [c.strip().lower() for c in l.columns]
        id_col, lang_col = l.columns[0], l.columns[1]
        l = l.rename(columns={id_col: "song", lang_col: "language"})
        top_langs = l["language"].value_counts().head(10).index.tolist()
        for lang in top_langs:
            col = f"lang_{lang.replace(' ', '_').lower()}"
            l[col] = (l["language"] == lang).astype(int)
            df = df.merge(l[["song", col]], on="song", how="left")

    print(f"  Content features added → shape {df.shape}")
    return df


# ── STEP 9 · SVD Embeddings ──────────────────────────────────

def build_svd_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    user_enc = LabelEncoder()
    song_enc = LabelEncoder()
    df["user_idx"] = user_enc.fit_transform(df["user"])
    df["song_idx"] = song_enc.fit_transform(df["song"])

    play_counts = df["us_play_count"].fillna(1).values
    mat = csr_matrix(
        (play_counts, (df["user_idx"].values, df["song_idx"].values)),
        shape=(df["user_idx"].max() + 1, df["song_idx"].max() + 1),
    )
    print(f"  Interaction matrix: {mat.shape} | non-zeros: {mat.nnz:,}")

    svd_user = TruncatedSVD(n_components=SVD_FACTORS, random_state=42)
    user_factors = svd_user.fit_transform(mat).astype("float32")

    svd_song = TruncatedSVD(n_components=SVD_FACTORS, random_state=42)
    song_factors = svd_song.fit_transform(mat.T).astype("float32")

    user_emb_cols = [f"user_emb_{i}" for i in range(SVD_FACTORS)]
    song_emb_cols = [f"song_emb_{i}" for i in range(SVD_FACTORS)]

    user_emb_df = pd.DataFrame(user_factors, columns=user_emb_cols)
    user_emb_df["user_idx"] = np.arange(len(user_emb_df))

    song_emb_df = pd.DataFrame(song_factors, columns=song_emb_cols)
    song_emb_df["song_idx"] = np.arange(len(song_emb_df))

    df = df.merge(user_emb_df, on="user_idx", how="left")
    df = df.merge(song_emb_df, on="song_idx", how="left")

    # Use context managers so file handles are always closed
    with open(os.path.join(MODEL_DIR, "user_encoder.pkl"), "wb") as f:
        pickle.dump(user_enc, f)
    with open(os.path.join(MODEL_DIR, "song_encoder.pkl"), "wb") as f:
        pickle.dump(song_enc, f)
    with open(os.path.join(MODEL_DIR, "svd_models.pkl"), "wb") as f:
        pickle.dump((svd_user, svd_song), f)

    print(
        f"  Explained variance — user: {svd_user.explained_variance_ratio_.sum():.3f} | "
        f"song: {svd_song.explained_variance_ratio_.sum():.3f}"
    )
    print(f"  SVD embeddings added → shape {df.shape}")
    return df


# ── STEP 10 · Feature Engineering Helpers ───────────────────

def drop_leaky_cols(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in LEAKY_COLS if c in df.columns]
    df = df.drop(columns=present)
    print(f"  Dropped leaky cols: {present} → shape {df.shape}")
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ── STEP 11 · Train / Validation / Test Split ───────────────

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["user"]))
    train_val = df.iloc[train_idx]
    test_df   = df.iloc[test_idx]

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=7)
    t_idx, v_idx = next(val_splitter.split(train_val, groups=train_val["user"]))
    train_df = train_val.iloc[t_idx]
    val_df   = train_val.iloc[v_idx]

    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(
            f"  {name:5}: {len(part):>8,} rows | "
            f"users: {part['user'].nunique():,} | "
            f"repeat rate: {part['label'].mean():.3f}"
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── STEP 12 · Train LightGBM ─────────────────────────────────

def train_model(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    feat_cols: List[str],
) -> lgb.Booster:
    neg = (train_df["label"] == 0).sum()
    pos = (train_df["label"] == 1).sum()
    params = {**LGB_PARAMS, "scale_pos_weight": neg / pos}
    print(f"  Class ratio (neg/pos): {neg/pos:.2f}")

    X_train = train_df[feat_cols].fillna(-1)
    X_val   = val_df[feat_cols].fillna(-1)
    dtrain  = lgb.Dataset(X_train, label=train_df["label"])
    dval    = lgb.Dataset(X_val,   label=val_df["label"], reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(30, verbose=True),
            lgb.log_evaluation(50),
        ],
    )

    model_path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, feat_cols), f)
    print(f"  Model saved → {model_path}")
    return model


# ── STEP 13 · Evaluate ───────────────────────────────────────

def evaluate(
    model:     lgb.Booster,
    test_df:   pd.DataFrame,
    feat_cols: List[str],
) -> pd.DataFrame:
    X_test = test_df[feat_cols].fillna(-1)
    probs  = model.predict(X_test)
    preds  = (probs >= 0.5).astype(int)

    print(f"  AUC-ROC           : {roc_auc_score(test_df['label'], probs):.4f}")
    print(f"  Average Precision : {average_precision_score(test_df['label'], probs):.4f}")
    print()
    print(classification_report(test_df["label"], preds, target_names=["no repeat", "repeat"]))

    # Feature importance plot
    fi = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feat_cols,
    ).sort_values(ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(fi.index, fi.values, color="#5563D4", edgecolor="white", linewidth=0.3)
    ax.set_title("Top 25 features by importance (gain)", fontsize=13)
    ax.set_xlabel("Importance (gain)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved → {plot_path}")

    # Only call plt.show() in an interactive notebook environment
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close()

    result_df = test_df[["user", "song", "label"]].copy()
    result_df["predicted_prob"] = probs
    out_path = os.path.join(MODEL_DIR, "test_predictions.csv")
    result_df.to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")
    return result_df


# ── STEP 14 · Recommend ──────────────────────────────────────

def recommend_for_user(
    user_id:     str,
    model:       lgb.Booster,
    feat_cols:   List[str],
    features_df: pd.DataFrame,
    top_n:       int = 10,
) -> pd.DataFrame:
    user_rows = features_df[features_df["user"] == user_id].copy()
    if user_rows.empty:
        print(f"  User '{user_id}' not found.")
        return pd.DataFrame()

    user_rows["repeat_prob"] = model.predict(user_rows[feat_cols].fillna(-1))
    return (
        user_rows
        .sort_values("repeat_prob", ascending=False)
        .head(top_n)[["user", "song", "repeat_prob", "label"]]
        .reset_index(drop=True)
    )


# ── MAIN ─────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "=" * 60)
    print("  SPOTIFY REPEAT PLAY PREDICTION — PIPELINE")
    print("=" * 60)

    # Load
    print("\n[1/8] Loading data...")
    data = load_all_data()
    history_raw = data["history"]

    # Preprocess
    print("\n[2/8] Preprocessing history...")
    history = preprocess_history(history_raw)

    # Target
    print("\n[3/8] Building target variable...")
    target_df = build_target(history, window_days=REPEAT_WINDOW_DAYS)

    # Features
    print("\n[4/8] Building collaborative features...")
    features_df = build_collaborative_features(history, target_df)

    print("\n[5/8] Building temporal features...")
    features_df = build_temporal_features(features_df)

    print("\n[6/8] Building content features...")
    features_df = build_content_features(
        features_df,
        data["genres"], data["tags"], data["metadata"], data["lang"],
    )

    # Embeddings
    print("\n[7/8] Building SVD embeddings...")
    features_df = build_svd_embeddings(features_df)
    features_df = drop_leaky_cols(features_df)

    # Downcast embedding columns from float64 → float32 to halve memory usage
    emb_cols = [c for c in features_df.columns if c.startswith(("user_emb_", "song_emb_"))]
    features_df[emb_cols] = features_df[emb_cols].astype("float32")

    # Save in chunks to avoid allocating the entire array at once
    feat_path = os.path.join(PROCESSED_DIR, "features.csv")
    chunk_size = 50_000
    for i, start in enumerate(range(0, len(features_df), chunk_size)):
        chunk = features_df.iloc[start: start + chunk_size]
        chunk.to_csv(feat_path, index=False, mode="w" if i == 0 else "a", header=(i == 0))
    print(f"  Feature matrix saved → {feat_path}  ({len(features_df):,} rows)")

    # Split
    print("\n[8/8] Splitting data...")
    train_df, val_df, test_df = split_data(features_df)
    feat_cols = get_feature_cols(features_df)
    print(f"  Feature count: {len(feat_cols)}")

    # Train
    print("\n[TRAINING] LightGBM...")
    model = train_model(train_df, val_df, feat_cols)

    # Evaluate
    print("\n[EVALUATION]")
    evaluate(model, test_df, feat_cols)

    # Recommend
    print("\n[RECOMMENDATIONS]")
    sample_user = features_df["user"].iloc[0]
    recs = recommend_for_user(sample_user, model, feat_cols, features_df, top_n=10)
    print(f"  Top 10 for user: {sample_user}")
    print(recs.to_string(index=False))

    print("\n✅ Pipeline complete.")
    return model, feat_cols, features_df


if __name__ == "__main__":
    model, feat_cols, features_df = run_pipeline()