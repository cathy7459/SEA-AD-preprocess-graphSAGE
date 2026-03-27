from __future__ import annotations

import pandas as pd


def standardize_donor_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]
    donor_col = next((col for col in out.columns if "donor" in col and "id" in col), None)
    if donor_col is None:
        donor_col = out.columns[0]
    out = out.rename(columns={donor_col: "donor_id"})
    out["donor_id"] = out["donor_id"].astype(str).str.strip()
    return out
