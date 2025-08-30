import re
import numpy as np
import pandas as pd

PRICE_MIN, PRICE_MAX = 40, 1000

def to_numeric_price(s: pd.Series) -> pd.Series:
    cleaned = s.astype(str).str.replace(r'[^0-9\.]+', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')

def extract_number(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)', str(x))
    return float(m.group(1)) if m else np.nan

def clean_core(df: pd.DataFrame) -> pd.DataFrame:
   
    col_price = next(c for c in df.columns if c.lower() in ["price","price_per_night","listing_price","nightly_price"])
    col_room  = next(c for c in df.columns if "room" in c.lower())
    col_nb    = next((c for c in df.columns if "neigh" in c.lower() or c.lower()=="city"), None)
    col_bed   = next((c for c in df.columns if "bedroom" in c.lower() or c.lower()=="beds"), None)
    col_bath  = next((c for c in df.columns if "bath" in c.lower()), None)

    work = pd.DataFrame()
    work["neighbourhood"] = df[col_nb].astype(str).str.strip() if col_nb else "Unknown"
    work["room_type"]     = df[col_room].astype(str).str.strip()
    work["price"]         = to_numeric_price(df[col_price])

    if col_bed:
        work["bedrooms"] = df[col_bed].apply(extract_number)
    else:
        work["bedrooms"] = np.nan

    if col_bath:
        work["bathrooms"] = df[col_bath].apply(extract_number)
    else:
        work["bathrooms"] = np.nan

    
    work = work.dropna(subset=["neighbourhood","room_type","price"])
   
    work = work[(work["price"]>=PRICE_MIN) & (work["price"]<=PRICE_MAX)]
   
    for c in ["bedrooms","bathrooms"]:
        work[c] = work[c].fillna(work[c].median(skipna=True))
    return work
