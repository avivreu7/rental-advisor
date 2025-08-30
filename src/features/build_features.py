import pandas as pd

def build_featured(work: pd.DataFrame) -> pd.DataFrame:
    featured = work.copy()
    for c in ["neighbourhood", "room_type"]:
        d = pd.get_dummies(featured[c], prefix=c, dummy_na=False)
        featured = pd.concat([featured.drop(columns=[c]), d], axis=1)
    return featured
