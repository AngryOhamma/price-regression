import pandas as pd
from pathlib import Path

def _read_csv_with_fallback(path):
    encodings = ["utf-8-sig", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="cp949", errors="replace")

def load_raw_data(data_path="data/raw"):
    files = list(Path(data_path).glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in data/raw")

    df_list = []
    for f in files:
        df_list.append(_read_csv_with_fallback(f))

    df = pd.concat(df_list, ignore_index=True)

    # KAMIS date is typically yyyymmdd as int/string -> parse explicitly
    if "가격등록일자" in df.columns:
        df["가격등록일자"] = pd.to_datetime(
            df["가격등록일자"].astype(str),
            format="%Y%m%d",
            errors="coerce"
        )

    return df