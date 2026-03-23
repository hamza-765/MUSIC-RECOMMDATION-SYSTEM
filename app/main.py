import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import load_artifacts

app = FastAPI(title="Spotify Repeat Predictor", version="1.0")

# ── Load model + features at startup ────────────────────────
try:
    artifacts = load_artifacts()
    model     = artifacts["model"]
    feat_cols = artifacts["feat_cols"]
    print("✅ Model loaded")
except Exception as e:
    print(f"⚠️  Model not loaded: {e}")
    model, feat_cols = None, []

# ── Load feature matrix for recommendations ──────────────────
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./processed")

try:
    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    print(f"✅ Features loaded | {len(features_df):,} rows")
except Exception as e:
    print(f"⚠️  Features not loaded: {e}")
    features_df = pd.DataFrame()


# ── Schemas ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    user_id:  str
    song_id:  str
    features: dict


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Spotify Repeat Predictor API"}


@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  model is not None,
        "feature_count": len(feat_cols),
        "users_in_db":   features_df["user"].nunique() if not features_df.empty else 0,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df   = pd.DataFrame([req.features]).fillna(-1)
    cols = [c for c in feat_cols if c in df.columns]

    if not cols:
        raise HTTPException(status_code=400, detail="No valid features provided")

    prob = float(model.predict(df[cols])[0])
    return {
        "user_id":     req.user_id,
        "song_id":     req.song_id,
        "repeat_prob": round(prob, 4),
        "will_repeat": prob >= 0.5,
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_n: int = 10):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if features_df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    # Look up user in feature matrix
    user_rows = features_df[features_df["user"] == user_id].copy()

    if user_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found. Check /users for valid IDs."
        )

    # Score all their songs
    user_rows["repeat_prob"] = model.predict(
        user_rows[feat_cols].fillna(-1)
    )

    # Return top N sorted by probability
    top = (
        user_rows
        .sort_values("repeat_prob", ascending=False)
        .head(top_n)[["user", "song", "repeat_prob", "label"]]
        .reset_index(drop=True)
    )

    return {
        "user_id": user_id,
        "recommendations": [
            {
                "song":        row["song"],
                "repeat_prob": round(row["repeat_prob"], 4),
                "label":       int(row["label"]),
            }
            for _, row in top.iterrows()
        ]
    }


@app.get("/users")
def list_users(limit: int = 20):
    """Returns sample valid user IDs you can use for recommendations."""
    if features_df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    users = features_df["user"].unique()[:limit].tolist()
    return {
        "total_users": features_df["user"].nunique(),
        "sample_ids":  users,
    }