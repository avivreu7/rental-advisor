# --- ensure src is importable when running from app/ ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------

import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from src.config import (CLEANED, MODEL_PKL, SCHEMA_JSON, OPENAI_API_KEY, OPENAI_MODEL)
from src.models.io import load_model
from src.explain.explain_fallback import explain_simple
from src.explain.llm_openai import build_client, explain_llm

st.set_page_config(page_title="SF Smart Rental Advisor", page_icon="🏠", layout="centered")
st.title("🏠 SF Smart Rental Advisor")
st.caption("Predict Airbnb nightly price for San Francisco + short natural-language explanation")

# Load data & model
if not Path(CLEANED).exists():
    st.error("Missing processed dataset: cleaned_data.csv. First run: `python scripts/build_dataset.py`")
    st.stop()

df = pd.read_csv(CLEANED)
if not Path(MODEL_PKL).exists():
    st.error("Missing trained model: model.pkl. Train with: `python scripts/train_model.py --model rf`")
    st.stop()

pipe = load_model(MODEL_PKL)
schema = {}
if Path(SCHEMA_JSON).exists():
    schema = json.loads(Path(SCHEMA_JSON).read_text(encoding="utf-8"))

# Select options from data
neighbourhoods = df["neighbourhood"].value_counts().index.tolist()
room_types = df["room_type"].value_counts().index.tolist()

col1, col2 = st.columns(2)
with col1:
    nb = st.selectbox("Neighbourhood", neighbourhoods, index=0)
    bedrooms = st.number_input(
        "Bedrooms", min_value=0.0, max_value=10.0,
        value=float(np.nanmedian(df["bedrooms"])) if "bedrooms" in df.columns else 1.0,
        step=0.5
    )
with col2:
    rt = st.selectbox("Room Type", room_types, index=0)
    bathrooms = st.number_input(
        "Bathrooms", min_value=0.0, max_value=10.0,
        value=float(np.nanmedian(df["bathrooms"])) if "bathrooms" in df.columns else 1.0,
        step=0.5
    )

if st.button("Predict"):
    X = pd.DataFrame([{
        "neighbourhood": nb,
        "room_type": rt,
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms)
    }])

    try:
        pred = float(pipe.predict(X)[0])
    except Exception as e:
        st.error(f"Model failed to predict: {e}")
        st.stop()

    st.success(f"💰 Predicted nightly price: ${pred:,.0f}")

    # Local stats
    subset = df[(df["neighbourhood"] == nb) & (df["room_type"] == rt)]["price"].dropna()
    ref_stats = {}
    if len(subset) >= 10:
        ref_stats = {
            "median": float(subset.median()),
            "p25": float(subset.quantile(0.25)),
            "p75": float(subset.quantile(0.75)),
            "n": int(len(subset))
        }
        st.write(
            f"**{nb} · {rt}** (n={ref_stats['n']}): "
            f"Median ${ref_stats['median']:,.0f}, IQR ${ref_stats['p25']:,.0f}–${ref_stats['p75']:,.0f}."
        )
        st.bar_chart(subset.reset_index(drop=True), height=160)
    else:
        st.info("Not enough matching examples to show a distribution for that neighbourhood/room type.")

    # LLM explanation (no temperature passed anywhere)
    st.subheader("🧠 Why this price?")
    client = build_client(OPENAI_API_KEY)
    explanation = ""
    if client and OPENAI_MODEL:
        try:
            explanation = explain_llm(client, OPENAI_MODEL, X.iloc[0].to_dict(), pred, ref_stats)
        except Exception as e:
            st.warning(f"LLM explanation failed ({e}); using fallback.")
    if not explanation:
        explanation = explain_simple(X.iloc[0].to_dict(), pred, ref_stats)

    st.write(explanation)

st.caption("Tip: retrain with `python scripts/train_model.py --model xgb` (or lgbm/cat) to compare performance.")
