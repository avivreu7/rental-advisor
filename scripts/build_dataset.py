# --- make src importable no matter where you run from ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------

from src.config import RAW_LISTINGS, CLEANED, FEATURED
from src.data.clean import clean_core
from src.features.build_features import build_featured
import pandas as pd

def main():
    # Load raw CSV → clean core columns → save processed datasets
    df = pd.read_csv(RAW_LISTINGS, low_memory=False)
    work = clean_core(df)

    CLEANED.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(CLEANED, index=False)

    # Optional: save one-hot featured table for inspection
    featured = build_featured(work)
    featured.to_csv(FEATURED, index=False)

    print("Saved:", CLEANED, "and", FEATURED)

if __name__ == "__main__":
    main()
