from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")  # loads if present

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"

RAW_LISTINGS = RAW_DIR / "listings.csv"
CLEANED = PROCESSED_DIR / "cleaned_data.csv"
FEATURED = PROCESSED_DIR / "featured_data.csv"

MODEL_PKL = MODELS_DIR / "model.pkl"
SCHEMA_JSON = MODELS_DIR / "model_schema.json"

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
