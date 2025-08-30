import pandas as pd
from src.config import RAW_LISTINGS, CLEANED, FEATURED

def load_raw() -> pd.DataFrame:
    """Load raw listings.csv from data/raw/"""
    return pd.read_csv(RAW_LISTINGS, low_memory=False)

def load_cleaned() -> pd.DataFrame:
    """Load cleaned dataset from data/processed/"""
    return pd.read_csv(CLEANED, low_memory=False)

def load_featured() -> pd.DataFrame:
    """Load featured dataset (one-hot) from data/processed/"""
    return pd.read_csv(FEATURED, low_memory=False)
