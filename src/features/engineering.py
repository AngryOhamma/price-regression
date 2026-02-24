import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assumes:
        df["date"] exists and is datetime
    """

    # Month / Week extraction
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    # Seasonal encoding (cyclical)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assumes:
        df["price"] exists
    """

    df = df.sort_values("date").copy()

    # Lag features
    df["lag_1"] = df["price"].shift(1)
    df["lag_2"] = df["price"].shift(2)
    df["lag_3"] = df["price"].shift(3)
    df["lag_7"] = df["price"].shift(7)

    # Rolling means (past-only)
    df["roll_mean_4"] = df["price"].shift(1).rolling(4).mean()
    df["roll_mean_8"] = df["price"].shift(1).rolling(8).mean()

    return df