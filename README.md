# KAMIS Wholesale Cabbage Price Prediction

## Overview
This project analyzes large-scale agricultural pricing data from the Korean Agricultural Marketing Information Service (KAMIS).
Approximately 3.8 million raw price records across multiple crop categories were processed and curated to construct a crop-specific time-series dataset for wholesale cabbage price modeling (2021–2023).

For clarity and reproducibility, key Korean column names were mapped to English aliases within the modeling pipeline (e.g., 가격등록일자 → `date`, 품목가격 → `price`).

## Objectives
- Predict 2023 wholesale cabbage prices using 2021–2022 data
- Mitigate multicollinearity in engineered time-series predictors
- Evaluate regularized regression models under noisy market conditions

## Data Pipeline
- Integrated six CSV files (2021–2023)
- Processed ~3.8M raw records and filtered to 3,835 wholesale cabbage samples  

## Feature Engineering
- Lag features and rolling window statistics
- Seasonal sinusoidal encodings (month/week sin-cos)
- 10 engineered predictors (before correlation filtering)

## Modeling & Evaluation
- Correlation filtering (|corr| > 0.95) to reduce multicollinearity  
  (10 predictors → 3 retained in the final fit)
- PCA + Ridge regression with 5-fold TimeSeries cross-validation
- Holdout evaluation on 2023

## Results (from `python main.py`)
- Train rows (2021–2022): 2,584  
- Test rows (2023): 1,243  
- Best model: PCA + Ridge (alpha ≈ 316, n_components = 3)
- 5-fold TimeSeries CV RMSE (log1p(price)): 0.197285
- 2023 Holdout RMSE (log1p(price)): 0.180629

## Project Structure
- `src/` — Modular preprocessing, feature engineering, and modeling code  
- `notebooks/` — Experimental analysis and visualizations  
- `main.py` — Executable entry point implementing the final selected model  

## How to Run
```bash
pip install -r requirements.txt
python main.py