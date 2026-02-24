import sys
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.data.loader import load_raw_data
from src.features.engineering import add_time_features, add_lag_rolling_features


# ============================================================
# Config
# ============================================================
CORR_THRESH = 0.95
N_SPLITS = 5
RIDGE_ALPHA_GRID = np.logspace(-3, 3, 13)
PCA_MAX_COMPONENTS = 10  # safety cap

# Korean -> English column aliases (presentation-friendly)
RENAME_MAP = {
    "가격등록일자": "date",
    "품목가격": "price",
    "품목명": "item_name",
    "품종명": "variety",
    "조사구분명": "market_type",
    "산물등급명": "grade",
    "할인가격여부": "discount_flag",
}


# ============================================================
# Helpers
# ============================================================
def drop_high_corr(X: pd.DataFrame, thresh: float):
    """Drop columns that have correlation > thresh with any earlier column."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > thresh)]
    return X.drop(columns=drop_cols), drop_cols


def train_pca_ridge_cv(X: pd.DataFrame, y: np.ndarray):
    """TimeSeries CV grid search for PCA + Ridge."""
    n_feat = X.shape[1]
    if n_feat < 2:
        raise ValueError(f"Too few features for PCA: {n_feat}")

    comps = list(range(2, min(PCA_MAX_COMPONENTS, n_feat) + 1))
    if not comps:
        raise ValueError("PCA component grid is empty. Check your feature count.")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("model", Ridge())
    ])

    grid = {
        "pca__n_components": comps,
        "model__alpha": RIDGE_ALPHA_GRID
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    gs = GridSearchCV(
        pipe,
        grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )
    gs.fit(X, y)
    return gs


def main():
    print("\n==============================")
    print("KAMIS Wholesale Cabbage Project")
    print("==============================")

    # --------------------------
    # Load
    # --------------------------
    print("\n[1/6] Loading raw data...")
    try:
        df = load_raw_data()
    except FileNotFoundError as e:
        print("\n[ERROR] Raw data files not found.")
        print("Place KAMIS CSVs in data/raw (note: data/raw is gitignored).")
        print("Details:", e)
        sys.exit(1)

    print(f"Raw shape: {df.shape}")

    # --------------------------
    # Rename & validate
    # --------------------------
    print("\n[2/6] Mapping columns to English aliases...")
    df = df.rename(columns=RENAME_MAP)

    required = ["date", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after rename: {missing}")

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    nat_count = int(df["date"].isna().sum())
    if nat_count > 0:
        print(f"[WARN] Found {nat_count} rows with invalid dates (NaT). They will be dropped later if needed.")

    # --------------------------
    # Filter (project scope)
    # --------------------------
    print("\n[3/6] Filtering to project scope...")
    # item_name == "배추" (cabbage), market_type == "도매" (wholesale),
    # grade == "상품" (top grade), discount_flag == "N" (no discount)
    filter_cols = ["item_name", "market_type", "grade", "discount_flag"]
    if all(c in df.columns for c in filter_cols):
        df = df[
            (df["item_name"] == "배추") &
            (df["market_type"] == "도매") &
            (df["grade"] == "상품") &
            (df["discount_flag"] == "N")
        ].copy()
    else:
        print("[WARN] Filter columns missing. Skipping strict filtering.")
        print("Missing:", [c for c in filter_cols if c not in df.columns])

    print(f"Filtered shape: {df.shape}")

    # Date sanity
    print("Date range:", df["date"].min(), "→", df["date"].max())
    year_counts = df["date"].dt.year.value_counts(dropna=False).sort_index()
    print("Year counts:")
    print(year_counts.to_string())

    # --------------------------
    # Feature Engineering
    # --------------------------
    print("\n[4/6] Feature engineering...")
    df = df.sort_values("date").copy()
    df = add_time_features(df)
    df = add_lag_rolling_features(df)

    feature_cols = [
        "week_sin", "week_cos", "month_sin", "month_cos",
        "lag_1", "lag_2", "lag_3", "lag_7",
        "roll_mean_4", "roll_mean_8"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "price"] + feature_cols).copy()

    # Split
    train_df = df[df["date"].dt.year <= 2022].copy()
    test_df = df[df["date"].dt.year == 2023].copy()

    print(f"Train rows (<=2022): {len(train_df)}")
    print(f"Test rows (2023):    {len(test_df)}")
    print(f"Feature count:       {len(feature_cols)}")

    # Prepare X/y (log1p stabilizes heavy-tailed prices)
    X0 = train_df[feature_cols].astype(float)
    y0 = np.log1p(train_df["price"].astype(float).values)

    # --------------------------
    # Multicollinearity mitigation
    # --------------------------
    print("\n[5/6] Correlation filtering + PCA + Ridge (TimeSeries CV)...")
    Xr, dropped = drop_high_corr(X0, CORR_THRESH)
    print(f"Correlation filter (|corr| > {CORR_THRESH})")
    print(f"  Dropped ({len(dropped)}): {dropped}")
    print(f"  Remaining features: {Xr.shape[1]}")

    gs = train_pca_ridge_cv(Xr, y0)

    best_rmse = float(-gs.best_score_)
    best_k = int(gs.best_params_["pca__n_components"])
    best_alpha = float(gs.best_params_["model__alpha"])

    print("\nBest model (CV): PCA + Ridge")
    print(f"  CV RMSE (log1p(price)): {best_rmse:.6f}")
    print(f"  PCA n_components:       {best_k}")
    print(f"  Ridge alpha:            {best_alpha:.6f}")

    # --------------------------
    # Holdout evaluation (2023)
    # --------------------------
    print("\n[6/6] Holdout evaluation on 2023...")
    if len(test_df) > 0:
        X_test0 = test_df[feature_cols].astype(float)
        # Drop same cols removed by correlation filtering
        X_test = X_test0.drop(columns=[c for c in dropped if c in X_test0.columns])
        y_test = np.log1p(test_df["price"].astype(float).values)

        pred = gs.best_estimator_.predict(X_test)
        rmse_test = float(np.sqrt(mean_squared_error(y_test, pred)))

        print(f"Holdout RMSE (2023, log1p(price)): {rmse_test:.6f}")
    else:
        print("[INFO] No 2023 rows found. Skipping holdout evaluation.")
        # -------------------------------------------------
    # Save results for reproducibility
    # -------------------------------------------------
    Path("reports").mkdir(exist_ok=True)

    results_path = Path("reports/results.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("KAMIS Wholesale Cabbage Project Results\n")
        f.write("========================================\n\n")
        f.write(f"Train rows (<=2022): {len(train_df)}\n")
        f.write(f"Test rows (2023): {len(test_df)}\n\n")
        f.write(f"Correlation threshold: {CORR_THRESH}\n")
        f.write(f"Dropped features: {dropped}\n")
        f.write(f"Remaining features: {Xr.shape[1]}\n\n")
        f.write("Best Model: PCA + Ridge\n")
        f.write(f"PCA n_components: {best_k}\n")
        f.write(f"Ridge alpha: {best_alpha}\n")
        f.write(f"CV RMSE (log1p): {best_rmse:.6f}\n")

        if len(test_df) > 0:
            f.write(f"Holdout RMSE 2023 (log1p): {rmse_test:.6f}\n")

    print(f"\nResults saved to: {results_path.resolve()}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()