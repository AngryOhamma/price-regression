# KAMIS Wholesale Cabbage Price Prediction

## Overview
This project analyzes large-scale agricultural pricing data from the Korean Agricultural Marketing Information Service (KAMIS).  
Approximately 3.8 million raw price records across multiple crop categories were processed and curated to construct a crop-specific time-series dataset for wholesale cabbage price modeling (2021–2023).

For clarity and reproducibility, key Korean column names were mapped to English aliases within the modeling pipeline (e.g., 가격등록일자 → `date`, 품목가격 → `price`).

## Objectives
- Predict 2023 wholesale cabbage prices using 2021–2022 data  
- Mitigate multicollinearity in engineered time-series predictors  
- Evaluate regularized regression models under noisy market conditions  

## Data Processing Pipeline
- Integrated six annual CSV files (2021–2023)  
- Processed ~3.8M raw records  
- Filtered to 3,835 wholesale cabbage observations  

## Feature Engineering
- Autoregressive lag features (1, 2, 3, 7-day)  
- Rolling window statistics (4-day, 8-day means)  
- Seasonal sinusoidal encodings (month/week sin-cos)  
- 10 engineered predictors prior to correlation filtering  

## Modeling Strategy
- Correlation filtering (|corr| > 0.95) to reduce multicollinearity  
  → 10 predictors reduced to 3 retained features  
- PCA-based dimensionality reduction  
- Ridge regression with 5-fold TimeSeries cross-validation  
- Out-of-sample holdout evaluation on 2023  

## Results (via `python main.py`)
- Train rows (2021–2022): 2,584  
- Test rows (2023): 1,243  
- Best model: **PCA + Ridge**  
  - α ≈ 316  
  - n_components = 3  
- 5-fold TimeSeries CV RMSE (log1p(price)): **0.197285**  
- 2023 Holdout RMSE (log1p(price)): **0.180629**

## Project Structure
- `src/` — Modular data loading, feature engineering, and modeling components  
- `notebooks/` — Exploratory analysis and experimental comparisons  
- `main.py` — Reproducible end-to-end training and evaluation pipeline  
- `reports/` — Automatically generated result summaries  

## Reproducibility

```bash
pip install -r requirements.txt
python main.py