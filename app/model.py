import os
import pickle

MODEL_DIR = os.getenv("MODEL_DIR", "./model_outputs")

# In-memory cache — loads once, reused on every request
_cache: dict = {}


def load_artifacts() -> dict:
    """Load all model artifacts from disk. Returns cached version after first load."""

    if _cache:
        return _cache

    files = {
        "lgbm":    "lgbm_model.pkl",
        "user_enc": "user_encoder.pkl",
        "song_enc": "song_encoder.pkl",
        "svd":     "svd_models.pkl",
    }

    # Check all files exist before loading anything
    missing = [
        name for name, fname in files.items()
        if not os.path.exists(os.path.join(MODEL_DIR, fname))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            f"Looking in: {os.path.abspath(MODEL_DIR)}\n"
            f"Run the pipeline first to generate model_outputs/"
        )

    # Load with context managers so handles are always closed
    with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
        model, feat_cols = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "user_encoder.pkl"), "rb") as f:
        user_enc = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "song_encoder.pkl"), "rb") as f:
        song_enc = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "svd_models.pkl"), "rb") as f:
        svd_user, svd_song = pickle.load(f)

    # Store in cache
    _cache["model"]     = model
    _cache["feat_cols"] = feat_cols
    _cache["user_enc"]  = user_enc
    _cache["song_enc"]  = song_enc
    _cache["svd_user"]  = svd_user
    _cache["svd_song"]  = svd_song

    print(f"✅ Artifacts loaded from {os.path.abspath(MODEL_DIR)}")
    print(f"   Features: {len(feat_cols)}")
    return _cache