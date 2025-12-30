import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import zipfile
import tempfile
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# Import all models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Try importing advanced models
try:
    import xgboost as xgb

    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except:
    HAS_LGB = False

try:
    import catboost as cb

    HAS_CAT = True
except:
    HAS_CAT = False

st.set_page_config(page_title="Airbnb ML Studio", page_icon="🏠", layout="wide")

st.title("🏠 Airbnb Price Prediction Studio")
st.markdown("**Full ML Pipeline with Advanced Data Cleaning & Detailed Logging**")
st.markdown("---")

# ============================================================================
# MODEL CONFIGURATIONS (Default and Tuned)
# ============================================================================

# Default models (original parameters)
DEFAULT_MODELS = {
    "Linear Regression": {"model": LinearRegression(), "category": "Linear", "scale": True},
    "Ridge Regression": {"model": Ridge(alpha=1.0), "category": "Linear", "scale": True},
    "Lasso Regression": {"model": Lasso(alpha=0.1), "category": "Linear", "scale": True},
    "ElasticNet": {"model": ElasticNet(alpha=0.1, l1_ratio=0.5), "category": "Linear", "scale": True},
    "Bayesian Ridge": {"model": BayesianRidge(), "category": "Linear", "scale": True},
    "Decision Tree": {"model": DecisionTreeRegressor(max_depth=15, random_state=42), "category": "Tree",
                      "scale": False},
    "Random Forest": {"model": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
                      "category": "Tree", "scale": False},
    "Extra Trees": {"model": ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
                    "category": "Tree", "scale": False},
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
        "category": "Boosting", "scale": False},
    "AdaBoost": {"model": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                 "category": "Boosting", "scale": False},
    "Bagging": {"model": BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1), "category": "Ensemble",
                "scale": False},
    "KNN": {"model": KNeighborsRegressor(n_neighbors=5, n_jobs=-1), "category": "Instance", "scale": True},
    "SVR": {"model": SVR(kernel='rbf', C=100, gamma='scale'), "category": "SVM", "scale": True},
    "Neural Network": {"model": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
                       "category": "Neural", "scale": True},
}

# Tuned models (optimized parameters for better accuracy)
TUNED_MODELS = {
    "Linear Regression": {"model": LinearRegression(), "category": "Linear", "scale": True},
    "Ridge Regression": {"model": Ridge(alpha=0.5), "category": "Linear", "scale": True},
    "Lasso Regression": {"model": Lasso(alpha=0.05), "category": "Linear", "scale": True},
    "ElasticNet": {"model": ElasticNet(alpha=0.05, l1_ratio=0.5), "category": "Linear", "scale": True},
    "Bayesian Ridge": {"model": BayesianRidge(), "category": "Linear", "scale": True},
    "Decision Tree": {"model": DecisionTreeRegressor(max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42),
                      "category": "Tree", "scale": False},
    "Random Forest": {"model": RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5,
                      min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1),
                      "category": "Tree", "scale": False},
    "Extra Trees": {"model": ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1),
                    "category": "Tree", "scale": False},
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                           min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42),
        "category": "Boosting", "scale": False},
    "AdaBoost": {"model": AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
                 "category": "Boosting", "scale": False},
    "Bagging": {"model": BaggingRegressor(n_estimators=200, max_samples=0.8, max_features=0.8, random_state=42, n_jobs=-1),
                "category": "Ensemble", "scale": False},
    "KNN": {"model": KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1), "category": "Instance", "scale": True},
    "SVR": {"model": SVR(kernel='rbf', C=150, gamma='scale', epsilon=0.1), "category": "SVM", "scale": True},
    "Neural Network": {"model": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                       learning_rate='adaptive', early_stopping=True, random_state=42),
                       "category": "Neural", "scale": True},
}

# Add XGBoost if available
if HAS_XGB:
    DEFAULT_MODELS["XGBoost"] = {
        "model": xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42, n_jobs=-1),
        "category": "Boosting", "scale": False}
    TUNED_MODELS["XGBoost"] = {
        "model": xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.03, min_child_weight=3,
                                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
                                  random_state=42, n_jobs=-1),
        "category": "Boosting", "scale": False}

# Add LightGBM if available
if HAS_LGB:
    DEFAULT_MODELS["LightGBM"] = {
        "model": lgb.LGBMRegressor(n_estimators=500, max_depth=12, learning_rate=0.05, num_leaves=50,
                                   subsample=0.8, random_state=42, n_jobs=-1, verbose=-1),
        "category": "Boosting", "scale": False}
    TUNED_MODELS["LightGBM"] = {
        "model": lgb.LGBMRegressor(n_estimators=1000, max_depth=10, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                   reg_alpha=0.1, reg_lambda=1, random_state=42, n_jobs=-1, verbose=-1),
        "category": "Boosting", "scale": False}

# Add CatBoost if available
if HAS_CAT:
    DEFAULT_MODELS["CatBoost"] = {
        "model": cb.CatBoostRegressor(iterations=500, depth=10, learning_rate=0.05, random_state=42, verbose=0),
        "category": "Boosting", "scale": False}
    TUNED_MODELS["CatBoost"] = {
        "model": cb.CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.03, l2_leaf_reg=3,
                                      random_strength=1, bagging_temperature=0.5, random_state=42, verbose=0),
        "category": "Boosting", "scale": False}

# Aggressive models (maximum iterations and fine-tuned parameters for best accuracy)
AGGRESSIVE_MODELS = {
    "Linear Regression": {"model": LinearRegression(), "category": "Linear", "scale": True},
    "Ridge Regression": {"model": Ridge(alpha=0.1), "category": "Linear", "scale": True},
    "Lasso Regression": {"model": Lasso(alpha=0.01), "category": "Linear", "scale": True},
    "ElasticNet": {"model": ElasticNet(alpha=0.01, l1_ratio=0.3), "category": "Linear", "scale": True},
    "Bayesian Ridge": {"model": BayesianRidge(), "category": "Linear", "scale": True},
    "Decision Tree": {"model": DecisionTreeRegressor(max_depth=25, min_samples_split=3, min_samples_leaf=1, random_state=42),
                      "category": "Tree", "scale": False},
    "Random Forest": {"model": RandomForestRegressor(n_estimators=800, max_depth=25, min_samples_split=3,
                      min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1),
                      "category": "Tree", "scale": False},
    "Extra Trees": {"model": ExtraTreesRegressor(n_estimators=800, max_depth=25, min_samples_split=3,
                    min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1),
                    "category": "Tree", "scale": False},
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=800, max_depth=8, learning_rate=0.02,
                                           min_samples_split=3, min_samples_leaf=1, subsample=0.85, random_state=42),
        "category": "Boosting", "scale": False},
    "AdaBoost": {"model": AdaBoostRegressor(n_estimators=300, learning_rate=0.03, random_state=42),
                 "category": "Boosting", "scale": False},
    "Bagging": {"model": BaggingRegressor(n_estimators=300, max_samples=0.9, max_features=0.9, random_state=42, n_jobs=-1),
                "category": "Ensemble", "scale": False},
    "KNN": {"model": KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1), "category": "Instance", "scale": True},
    "SVR": {"model": SVR(kernel='rbf', C=200, gamma='scale', epsilon=0.05), "category": "SVM", "scale": True},
    "Neural Network": {"model": MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000,
                       learning_rate='adaptive', early_stopping=True, random_state=42),
                       "category": "Neural", "scale": True},
}

# Add XGBoost aggressive if available
if HAS_XGB:
    AGGRESSIVE_MODELS["XGBoost"] = {
        "model": xgb.XGBRegressor(n_estimators=1500, max_depth=10, learning_rate=0.02, min_child_weight=2,
                                  subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5,
                                  random_state=42, n_jobs=-1),
        "category": "Boosting", "scale": False}

# Add LightGBM aggressive if available
if HAS_LGB:
    AGGRESSIVE_MODELS["LightGBM"] = {
        "model": lgb.LGBMRegressor(n_estimators=1500, max_depth=12, learning_rate=0.02, num_leaves=50,
                                   min_child_samples=10, subsample=0.85, colsample_bytree=0.85,
                                   reg_alpha=0.05, reg_lambda=0.5, random_state=42, n_jobs=-1, verbose=-1),
        "category": "Boosting", "scale": False}

# Add CatBoost aggressive if available
if HAS_CAT:
    AGGRESSIVE_MODELS["CatBoost"] = {
        "model": cb.CatBoostRegressor(iterations=1500, depth=10, learning_rate=0.02, l2_leaf_reg=1,
                                      random_strength=0.5, bagging_temperature=0.3, random_state=42, verbose=0),
        "category": "Boosting", "scale": False}

# Default to DEFAULT_MODELS (will be switched based on user selection)
ALL_MODELS = DEFAULT_MODELS.copy()


# ============================================================================
# CLEANING FUNCTIONS WITH LOGGING
# ============================================================================

class CleaningLogger:
    def __init__(self):
        self.logs = []
        self.stats = {}
        self.examples = {}

    def log(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"time": timestamp, "message": message, "level": level})

    def add_stat(self, category, key, value):
        if category not in self.stats:
            self.stats[category] = {}
        self.stats[category][key] = value

    def add_example(self, category, before, after):
        if category not in self.examples:
            self.examples[category] = []
        self.examples[category].append({"before": before, "after": after})

    def get_logs(self):
        return self.logs

    def get_stats(self):
        return self.stats

    def get_examples(self):
        return self.examples


def clean_price_with_log(price_series, logger):
    """Clean price column with detailed logging"""
    original = price_series.copy()

    # Count different formats
    has_dollar = price_series.astype(str).str.contains(r'\$', na=False).sum()
    has_comma = price_series.astype(str).str.contains(r',', na=False).sum()

    logger.log(f"Found {has_dollar:,} values with '$' symbol")
    logger.log(f"Found {has_comma:,} values with ',' separator")

    # Clean
    cleaned = price_series.astype(str).replace(r'[\$,]', '', regex=True)
    cleaned = pd.to_numeric(cleaned, errors='coerce')

    # Stats
    valid = cleaned.notna().sum()
    invalid = cleaned.isna().sum()

    logger.add_stat("Price Cleaning", "Original count", len(price_series))
    logger.add_stat("Price Cleaning", "Valid after cleaning", valid)
    logger.add_stat("Price Cleaning", "Invalid (NaN)", invalid)
    logger.add_stat("Price Cleaning", "Had $ symbol", has_dollar)
    logger.add_stat("Price Cleaning", "Had comma separator", has_comma)

    # Examples
    sample_idx = original[original.astype(str).str.contains(r'\$', na=False)].head(3).index
    for idx in sample_idx:
        logger.add_example("Price", str(original[idx]), str(cleaned[idx]))

    logger.log(f"✅ Price cleaning complete: {valid:,} valid, {invalid:,} invalid", "success")

    return cleaned


def clean_percentage_with_log(pct_series, col_name, logger):
    """Clean percentage column with detailed logging"""
    original = pct_series.copy()

    has_percent = pct_series.astype(str).str.contains(r'%', na=False).sum()
    logger.log(f"Column '{col_name}': Found {has_percent:,} values with '%' symbol")

    cleaned = pct_series.astype(str).replace(r'[%]', '', regex=True)
    cleaned = pd.to_numeric(cleaned, errors='coerce') / 100

    valid = cleaned.notna().sum()
    invalid = cleaned.isna().sum()

    logger.add_stat(f"Percentage: {col_name}", "Valid", valid)
    logger.add_stat(f"Percentage: {col_name}", "Invalid", invalid)
    logger.add_stat(f"Percentage: {col_name}", "Mean", f"{cleaned.mean():.2%}" if valid > 0 else "N/A")

    # Examples
    sample_idx = original[original.astype(str).str.contains(r'%', na=False)].head(2).index
    for idx in sample_idx:
        logger.add_example(f"Percentage ({col_name})", str(original[idx]),
                           f"{cleaned[idx]:.2f}" if pd.notna(cleaned[idx]) else "NaN")

    return cleaned


def validate_coordinates_with_log(df, logger, lat_col='latitude', lon_col='longitude'):
    """Validate coordinates with detailed logging"""

    total = len(df)

    # Check for missing
    missing_lat = df[lat_col].isna().sum()
    missing_lon = df[lon_col].isna().sum()

    logger.log(f"Missing latitude: {missing_lat:,}")
    logger.log(f"Missing longitude: {missing_lon:,}")

    # Valid ranges
    valid_lat = (df[lat_col] >= -90) & (df[lat_col] <= 90)
    valid_lon = (df[lon_col] >= -180) & (df[lon_col] <= 180)
    valid_coords = valid_lat & valid_lon

    invalid_lat = (~valid_lat).sum()
    invalid_lon = (~valid_lon).sum()

    # Geographic bounds check (North America)
    in_na_lat = (df[lat_col] >= 20) & (df[lat_col] <= 70)
    in_na_lon = (df[lon_col] >= -170) & (df[lon_col] <= -50)
    in_north_america = in_na_lat & in_na_lon
    outside_na = (~in_north_america & valid_coords).sum()

    # Stats
    logger.add_stat("Coordinate Validation", "Total rows", total)
    logger.add_stat("Coordinate Validation", "Missing latitude", missing_lat)
    logger.add_stat("Coordinate Validation", "Missing longitude", missing_lon)
    logger.add_stat("Coordinate Validation", "Invalid latitude (out of -90 to 90)", invalid_lat)
    logger.add_stat("Coordinate Validation", "Invalid longitude (out of -180 to 180)", invalid_lon)
    logger.add_stat("Coordinate Validation", "Outside North America", outside_na)
    logger.add_stat("Coordinate Validation", "Valid coordinates", valid_coords.sum())

    # Range info
    logger.add_stat("Coordinate Validation", "Latitude range", f"{df[lat_col].min():.4f} to {df[lat_col].max():.4f}")
    logger.add_stat("Coordinate Validation", "Longitude range", f"{df[lon_col].min():.4f} to {df[lon_col].max():.4f}")

    logger.log(f"✅ Coordinates validated: {valid_coords.sum():,} valid, {(~valid_coords).sum():,} invalid", "success")

    return valid_coords


def remove_duplicates_with_log(df, subset_cols, logger, keep='first'):
    """Remove duplicates with detailed logging"""

    original_len = len(df)

    # Find duplicates
    duplicates = df.duplicated(subset=subset_cols, keep=False)
    dup_count = duplicates.sum()

    # How many unique duplicate groups
    if dup_count > 0:
        dup_groups = df[duplicates].groupby(subset_cols).size()
        num_groups = len(dup_groups)
        max_dups = dup_groups.max()

        logger.log(f"Found {dup_count:,} duplicate rows in {num_groups:,} groups")
        logger.log(f"Largest duplicate group: {max_dups} identical rows")
    else:
        logger.log("No duplicates found")

    # Remove
    df_clean = df.drop_duplicates(subset=subset_cols, keep=keep)
    removed = original_len - len(df_clean)

    logger.add_stat("Duplicate Removal", "Original rows", original_len)
    logger.add_stat("Duplicate Removal", "Duplicates found", dup_count)
    logger.add_stat("Duplicate Removal", "Rows removed", removed)
    logger.add_stat("Duplicate Removal", "Final rows", len(df_clean))
    logger.add_stat("Duplicate Removal", "Key columns", ", ".join(subset_cols))

    logger.log(f"✅ Removed {removed:,} duplicate rows", "success")

    return df_clean, removed


def handle_missing_with_log(df, columns, logger, strategy='median'):
    """Handle missing values with detailed logging"""

    missing_report = []
    total_filled = 0

    for col in columns:
        if col not in df.columns:
            continue

        missing = df[col].isna().sum()
        if missing > 0:
            pct = missing / len(df) * 100

            if strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0

            df[col] = df[col].fillna(fill_value)
            total_filled += missing

            missing_report.append({
                'Column': col,
                'Missing': missing,
                'Percent': f"{pct:.1f}%",
                'Filled with': f"{fill_value:.2f}" if isinstance(fill_value, float) else str(fill_value)
            })

            logger.log(f"Column '{col}': filled {missing:,} missing ({pct:.1f}%) with {strategy}={fill_value:.2f}")

    logger.add_stat("Missing Values", "Columns processed", len(columns))
    logger.add_stat("Missing Values", "Total values filled", total_filled)
    logger.add_stat("Missing Values", "Strategy used", strategy)

    return df, missing_report, total_filled


def detect_outliers_with_log(series, logger, method='percentile', **kwargs):
    """Detect outliers with detailed logging"""

    original_stats = {
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'median': series.median(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75)
    }

    logger.log(f"Outlier detection using {method} method")
    logger.log(f"Original stats: mean=${original_stats['mean']:.2f}, std=${original_stats['std']:.2f}")
    logger.log(f"Range: ${original_stats['min']:.2f} to ${original_stats['max']:.2f}")

    if method == 'percentile':
        lower_pct = kwargs.get('lower', 0.03)
        upper_pct = kwargs.get('upper', 0.97)
        lower_bound = series.quantile(lower_pct)
        upper_bound = series.quantile(upper_pct)
        outliers = (series < lower_bound) | (series > upper_bound)
        method_desc = f"Percentile ({lower_pct * 100:.0f}%-{upper_pct * 100:.0f}%)"

    elif method == 'iqr':
        multiplier = kwargs.get('multiplier', 1.5)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        method_desc = f"IQR ({multiplier}x)"

    else:  # zscore
        threshold = kwargs.get('threshold', 3)
        mean = series.mean()
        std = series.std()
        z_scores = np.abs((series - mean) / std)
        outliers = z_scores > threshold
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        method_desc = f"Z-Score ({threshold}σ)"

    outlier_count = outliers.sum()
    outlier_pct = outlier_count / len(series) * 100

    # Outlier analysis
    low_outliers = (series < lower_bound).sum()
    high_outliers = (series > upper_bound).sum()

    logger.add_stat("Outlier Detection", "Method", method_desc)
    logger.add_stat("Outlier Detection", "Lower bound", f"${lower_bound:.2f}")
    logger.add_stat("Outlier Detection", "Upper bound", f"${upper_bound:.2f}")
    logger.add_stat("Outlier Detection", "Total outliers", f"{outlier_count:,} ({outlier_pct:.1f}%)")
    logger.add_stat("Outlier Detection", "Low outliers (below bound)", low_outliers)
    logger.add_stat("Outlier Detection", "High outliers (above bound)", high_outliers)

    # Examples of outliers
    if outlier_count > 0:
        outlier_values = series[outliers].head(5).tolist()
        logger.add_example("Outliers", "Sample outlier values", [f"${v:.2f}" for v in outlier_values])

    logger.log(f"✅ Found {outlier_count:,} outliers ({outlier_pct:.1f}%): {low_outliers:,} low, {high_outliers:,} high",
               "success")

    return outliers, lower_bound, upper_bound


def analyze_data_quality(df, logger):
    """Analyze overall data quality"""

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_pct = missing_cells / total_cells * 100

    # Column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Missing by column
    missing_by_col = df.isna().sum()
    cols_with_missing = (missing_by_col > 0).sum()
    most_missing_col = missing_by_col.idxmax() if missing_by_col.max() > 0 else "None"
    most_missing_count = missing_by_col.max()

    logger.add_stat("Data Quality Overview", "Total rows", f"{df.shape[0]:,}")
    logger.add_stat("Data Quality Overview", "Total columns", df.shape[1])
    logger.add_stat("Data Quality Overview", "Numeric columns", len(numeric_cols))
    logger.add_stat("Data Quality Overview", "Text columns", len(text_cols))
    logger.add_stat("Data Quality Overview", "Total cells", f"{total_cells:,}")
    logger.add_stat("Data Quality Overview", "Missing cells", f"{missing_cells:,} ({missing_pct:.1f}%)")
    logger.add_stat("Data Quality Overview", "Columns with missing data", cols_with_missing)
    logger.add_stat("Data Quality Overview", "Most missing column", f"{most_missing_col} ({most_missing_count:,})")

    return {
        'total_rows': df.shape[0],
        'total_cols': df.shape[1],
        'missing_pct': missing_pct,
        'numeric_cols': len(numeric_cols),
        'text_cols': len(text_cols)
    }


# Session state
for key in ['listings', 'calendar', 'reviews', 'neighbourhoods', 'merged_data',
            'X_train', 'X_test', 'y_train', 'y_test', 'X_train_scaled', 'X_test_scaled',
            'scaler', 'features', 'trained_models', 'model_results', 'cleaning_logger',
            'le_room', 'le_neigh', 'le_prop',
            'all_listings', 'all_calendar', 'all_reviews', 'all_neighbourhoods',
            'dataset_sources', 'files_found']:
    if key not in st.session_state:
        st.session_state[key] = None

# Initialize data_combined flag
if 'data_combined' not in st.session_state:
    st.session_state.data_combined = False

if st.session_state.trained_models is None:
    st.session_state.trained_models = {}
if st.session_state.model_results is None:
    st.session_state.model_results = {}

# Tabs
tabs = st.tabs(["1️⃣ Upload", "2️⃣ Cleaning", "3️⃣ Process", "4️⃣ Features",
                "5️⃣ Models", "6️⃣ Training", "7️⃣ Results", "8️⃣ Predict"])

# ============================================================================
# TAB 1: UPLOAD
# ============================================================================
with tabs[0]:
    st.header("1️⃣ Data Upload")

    st.markdown("""
    ### 📁 Upload ZIP file(s) containing:
    | File | Description | Required |
    |------|-------------|----------|
    | `listings.csv/.xls` | Property details, prices, host info | ✅ Yes |
    | `calendar.csv/.xls` | 365 days of prices & availability | ⭐ Recommended |
    | `reviews.csv/.xls` | Guest comments and dates | ⭐ Recommended |
    | `neighbourhoods.csv/.xls` | Geographic boundaries | Optional |

    **💡 Multiple Datasets:** You can upload multiple ZIP files (e.g., December + March data) and they will be automatically combined.
    """)

    # Required Features Section
    with st.expander("📋 **Features Used in Model** (Click to expand)", expanded=False):
        st.markdown("""
        ### Required & Recommended Columns by Data Source

        The model uses these columns for training. Missing columns will be handled automatically, but having more columns improves prediction accuracy.

        ---
        #### 🏠 **listings.csv** (Required)

        | Category | Columns | Purpose |
        |----------|---------|---------|
        | **Target** | `price` | The price to predict (will be cleaned of $, commas) |
        | **Property** | `accommodates`, `bedrooms`, `beds`, `bathrooms` | Core property size features |
        | **Location** | `latitude`, `longitude`, `neighbourhood_cleansed` | Geographic features |
        | **Type** | `room_type`, `property_type` | Listing classification |
        | **Host** | `host_is_superhost`, `host_identity_verified`, `host_has_profile_pic` | Host quality indicators |
        | **Host Stats** | `host_listings_count`, `host_response_rate`, `host_acceptance_rate` | Host activity metrics |
        | **Booking** | `instant_bookable`, `minimum_nights` | Booking policies |
        | **Content** | `amenities`, `description`, `name` | Text features for NLP |
        | **Reviews** | `review_scores_rating`, `review_scores_cleanliness`, `review_scores_location`, `review_scores_value` | Review scores |
        | **Review Stats** | `number_of_reviews`, `reviews_per_month` | Review volume metrics |

        ---
        #### 📅 **calendar.csv** (Recommended)

        | Column | Purpose |
        |--------|---------|
        | `listing_id` | Links to listings |
        | `date` | Date of availability |
        | `price` | Daily price (for price statistics) |
        | `available` | Availability status (t/f) |

        *Used to calculate: price mean/std/min/max, availability rate, seasonal patterns, demand indicators*

        ---
        #### ⭐ **reviews.csv** (Recommended)

        | Column | Purpose |
        |--------|---------|
        | `listing_id` | Links to listings |
        | `date` | Review date |
        | `comments` | Review text for sentiment analysis |

        *Used to calculate: sentiment scores, review recency, comment length statistics*

        ---
        #### 📍 **neighbourhoods.csv** (Optional)

        | Column | Purpose |
        |--------|---------|
        | `neighbourhood` | Neighbourhood name |
        | `neighbourhood_group` | Neighbourhood grouping |
        | `geometry` | Polygon coordinates |

        *Used to calculate: area, perimeter, centroid, neighbourhood statistics*

        ---
        ### 🔧 Derived Features (Auto-Generated)

        The model automatically creates **80+ features** including:
        - **Interaction features**: `bedrooms × location_score`, `accommodates × superhost`, etc.
        - **Location clusters**: K-Means clustering of lat/long coordinates
        - **Text features**: Amenity parsing (wifi, parking, kitchen, AC, etc.)
        - **Polynomial features**: Squared and log transforms
        - **Ratio features**: beds_per_bedroom, capacity_score, etc.
        - **Sentiment features**: Positive/negative word counts from reviews
        """)

        st.info("💡 **Tip:** The more columns you have from the list above, the better your model's prediction accuracy will be!")

    # Manual Prediction Input Fields Section
    with st.expander("🔮 **Manual Prediction Input Fields** (Click to expand)", expanded=False):
        st.markdown("""
        ### Input Fields for Manual Price Prediction

        When using the **"Enter Property Details"** feature to predict prices manually, you'll be asked to provide the following information:

        ---
        #### 🏠 Property Details

        | Field | Type | Options/Range | Description |
        |-------|------|---------------|-------------|
        | Room Type | Dropdown | Entire home/apt, Private room, Shared room, Hotel room | Type of accommodation |
        | Bedrooms | Dropdown | 0-10 | Number of bedrooms |
        | Bathrooms | Dropdown | 0-5 (0.5 increments) | Number of bathrooms |
        | Beds | Dropdown | 0-10 | Number of beds |
        | Accommodates | Dropdown | 1-16 | Maximum number of guests |
        | Minimum Nights | Dropdown | 1, 2, 3, 4, 5, 7, 14, 30 | Minimum stay requirement |
        | Neighbourhood | Dropdown | *From your data* | Location neighbourhood |
        | Instant Bookable | Dropdown | Yes / No | Can guests book instantly? |
        | Superhost | Dropdown | Yes / No | Are you a Superhost? |

        ---
        #### 📍 Location

        | Field | Type | Range | Description |
        |-------|------|-------|-------------|
        | Latitude | Number | -90 to 90 | Geographic latitude |
        | Longitude | Number | -180 to 180 | Geographic longitude |

        ---
        #### 👤 Host & Listing Details

        | Field | Type | Options/Range | Description |
        |-------|------|---------------|-------------|
        | Host Identity Verified | Dropdown | Yes / No | Is host identity verified? |
        | Total Listings by Host | Number | 1-100 | Number of listings managed |
        | Number of Amenities | Slider | 0-100 | Total amenities offered |
        | Description Length | Slider | 0-2000 | Character count of description |
        | Luxury Keywords | Dropdown | Yes / No | Has luxury terms in title? |

        ---
        #### ⭐ Review Scores

        | Field | Type | Range | Description |
        |-------|------|-------|-------------|
        | Overall Rating | Slider | 0.0-5.0 | Average review rating |
        | Cleanliness | Slider | 0.0-5.0 | Cleanliness score |
        | Location | Slider | 0.0-5.0 | Location score |
        | Value | Slider | 0.0-5.0 | Value for money score |
        | Number of Reviews | Number | 0-1000 | Total review count |
        | Reviews per Month | Number | 0.0-30.0 | Average monthly reviews |

        ---
        #### 📅 Prediction Settings

        | Field | Type | Options | Description |
        |-------|------|---------|-------------|
        | Month | Dropdown | January - December | Target month for seasonal pricing |
        | Models | Multi-select | *Trained models* | Models to use for prediction |
        """)

        st.info("💡 **Tip:** The more accurate your inputs, the better the price prediction! Review scores and amenities have significant impact on pricing.")

    uploaded_zips = st.file_uploader("📁 Upload ZIP File(s)", type=['zip'], accept_multiple_files=True)

    if uploaded_zips is not None and len(uploaded_zips) > 0:
        # Check if these are the same files we already processed
        current_file_names = sorted([z.name for z in uploaded_zips])
        previous_file_names = sorted(st.session_state.get('dataset_sources') or [])

        # Only process if files are different
        if current_file_names != previous_file_names:
            with st.spinner(f"📦 Extracting {len(uploaded_zips)} file(s)..."):
                # Temporary storage for all dataframes from all ZIPs
                all_listings = []
                all_calendar = []
                all_reviews = []
                all_neighbourhoods = []
                files_found = []
                dataset_sources = []

                for zip_idx, uploaded_zip in enumerate(uploaded_zips):
                    zip_name = uploaded_zip.name
                    dataset_sources.append(zip_name)

                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "data.zip")
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_zip.read())
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(tmpdir)

                    for root, dirs, files in os.walk(tmpdir):
                        for f in files:
                            fpath = os.path.join(root, f)
                            flow = f.lower()
                            if 'listing' in flow and flow.endswith(('.xls', '.csv')):
                                df = pd.read_csv(fpath)
                                df['_source_dataset'] = zip_name  # Track source
                                all_listings.append(df)
                                files_found.append(('listings', f, len(df), zip_name))
                            elif 'calendar' in flow and flow.endswith(('.xls', '.csv')):
                                df = pd.read_csv(fpath)
                                df['_source_dataset'] = zip_name
                                all_calendar.append(df)
                                files_found.append(('calendar', f, len(df), zip_name))
                            elif 'review' in flow and flow.endswith(('.xls', '.csv')):
                                df = pd.read_csv(fpath)
                                df['_source_dataset'] = zip_name
                                all_reviews.append(df)
                                files_found.append(('reviews', f, len(df), zip_name))
                            elif 'neighbour' in flow and flow.endswith(('.xls', '.csv')):
                                df = pd.read_csv(fpath)
                                df['_source_dataset'] = zip_name
                                all_neighbourhoods.append(df)
                                files_found.append(('neighbourhoods', f, len(df), zip_name))

                # Store individual dataframes for combining in Cleaning tab
                st.session_state.all_listings = all_listings
                st.session_state.all_calendar = all_calendar
                st.session_state.all_reviews = all_reviews
                st.session_state.all_neighbourhoods = all_neighbourhoods
                st.session_state.dataset_sources = dataset_sources
                st.session_state.files_found = files_found

                # Clear any previously combined data
                st.session_state.listings = None
                st.session_state.calendar = None
                st.session_state.reviews = None
                st.session_state.neighbourhoods = None
                st.session_state.data_combined = False

            st.success(f"✅ New files extracted from {len(uploaded_zips)} dataset(s)!")

        # Show files already loaded (from session state)
        files_found = st.session_state.get('files_found') or []
        dataset_sources = st.session_state.get('dataset_sources') or []
        all_listings = st.session_state.get('all_listings') or []
        all_calendar = st.session_state.get('all_calendar') or []
        all_reviews = st.session_state.get('all_reviews') or []
        all_neighbourhoods = st.session_state.get('all_neighbourhoods') or []

        if st.session_state.get('data_combined'):
            st.info("✅ **Data already combined.** Go to the **Cleaning** tab to start or continue cleaning.")
        else:
            st.info("📌 **Next Step:** Go to the **Cleaning** tab to combine and clean the data.")

        # Show files found per dataset
        st.subheader("📋 Files Loaded")
        for dtype, fname, count, source in files_found:
            source_label = f" [{source}]" if len(dataset_sources) > 1 else ""
            st.write(f"  ✅ **{dtype}**: `{fname}` ({count:,} rows){source_label}")

        # Metrics (individual file totals - will be combined in Cleaning)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if all_listings:
                total_rows = sum(len(df) for df in all_listings)
                st.metric("🏠 Listings", f"{total_rows:,}")
                st.caption(f"{len(all_listings)} file(s) | Ready to combine")
            else:
                st.error("❌ Missing")
        with col2:
            if all_calendar:
                total_rows = sum(len(df) for df in all_calendar)
                st.metric("📅 Calendar", f"{total_rows:,}")
                st.caption(f"{len(all_calendar)} file(s) | Ready to combine")
            else:
                st.warning("⚠️ Not found")
        with col3:
            if all_reviews:
                total_rows = sum(len(df) for df in all_reviews)
                st.metric("⭐ Reviews", f"{total_rows:,}")
                st.caption(f"{len(all_reviews)} file(s) | Ready to combine")
            else:
                st.warning("⚠️ Not found")
        with col4:
            if all_neighbourhoods:
                total_rows = sum(len(df) for df in all_neighbourhoods)
                st.metric("📍 Neighbourhoods", f"{total_rows:,}")
                st.caption(f"{len(all_neighbourhoods)} file(s) | Ready to combine")
            else:
                st.warning("⚠️ Not found")

        # Data preview (show first file from each type)
        if st.checkbox("📋 Preview Raw Data"):
            preview_options = []
            if all_listings:
                preview_options.append('listings')
            if all_calendar:
                preview_options.append('calendar')
            if all_reviews:
                preview_options.append('reviews')
            if all_neighbourhoods:
                preview_options.append('neighbourhoods')

            if preview_options:
                preview_choice = st.selectbox("Select dataset:", preview_options)
                data_list = st.session_state.get(f'all_{preview_choice}', [])
                if data_list:
                    st.write(f"**Showing first file** (Total: {len(data_list)} file(s))")
                    data = data_list[0]
                    st.write(f"**Shape:** {data.shape[0]:,} rows × {data.shape[1]} columns")
                    st.dataframe(data.head(10), use_container_width=True)

                    # Column info
                    with st.expander("📊 Column Information"):
                        col_info = pd.DataFrame({
                            'Column': data.columns,
                            'Type': data.dtypes.astype(str),
                            'Non-Null': data.notna().sum(),
                            'Null': data.isna().sum(),
                            'Null %': (data.isna().sum() / len(data) * 100).round(1)
                        })
                        st.dataframe(col_info, use_container_width=True)

        # ====================================================================
        # 2.1 Dataset Description (Preview - full EDA after combining in Cleaning)
        # ====================================================================
        st.markdown("---")
        st.subheader("2.1 Dataset Description (Preview)")
        st.caption("📌 This shows the first uploaded file. Full combined data analysis available after combining in Cleaning tab.")

        all_listings_list = st.session_state.get('all_listings', [])
        if all_listings_list:
            listings_df = all_listings_list[0]  # Use first file for preview

            st.markdown("#### Dataset Size")
            size_col1, size_col2, size_col3 = st.columns(3)
            with size_col1:
                total_rows = len(listings_df)
                st.metric("Total Listings", f"{total_rows:,}")
            with size_col2:
                total_cols = len(listings_df.columns)
                st.metric("Total Features", f"{total_cols}")
            with size_col3:
                memory_mb = listings_df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Memory Usage", f"{memory_mb:.2f} MB")

            st.markdown("#### Field Categories")
            # Categorize columns by type
            numeric_cols = listings_df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = listings_df.select_dtypes(include=['object']).columns.tolist()
            date_cols = [c for c in listings_df.columns if 'date' in c.lower() or 'time' in c.lower()]

            field_col1, field_col2, field_col3 = st.columns(3)
            with field_col1:
                st.write(f"**Numeric Fields:** {len(numeric_cols)}")
                with st.expander("View numeric columns"):
                    st.write(", ".join(numeric_cols[:20]) + ("..." if len(numeric_cols) > 20 else ""))
            with field_col2:
                st.write(f"**Text Fields:** {len(text_cols)}")
                with st.expander("View text columns"):
                    st.write(", ".join(text_cols[:20]) + ("..." if len(text_cols) > 20 else ""))
            with field_col3:
                st.write(f"**Date-related Fields:** {len(date_cols)}")
                with st.expander("View date columns"):
                    st.write(", ".join(date_cols) if date_cols else "None detected")

            st.markdown("#### Data Completeness")
            completeness = (listings_df.notna().sum() / len(listings_df) * 100).round(1)
            avg_completeness = completeness.mean()
            fully_complete = (completeness == 100).sum()
            highly_complete = ((completeness >= 90) & (completeness < 100)).sum()
            partial = ((completeness >= 50) & (completeness < 90)).sum()
            sparse = (completeness < 50).sum()

            comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
            with comp_col1:
                st.metric("Avg Completeness", f"{avg_completeness:.1f}%")
            with comp_col2:
                st.metric("Fully Complete", f"{fully_complete} cols")
            with comp_col3:
                st.metric("Highly Complete (≥90%)", f"{highly_complete} cols")
            with comp_col4:
                st.metric("Sparse (<50%)", f"{sparse} cols")

            # Show most incomplete columns
            if sparse > 0:
                with st.expander("⚠️ Columns with low completeness (<50%)"):
                    low_complete = completeness[completeness < 50].sort_values()
                    for col, pct in low_complete.items():
                        st.write(f"- `{col}`: {pct}% complete")

        # ====================================================================
        # 2.2 Exploratory Data Analysis (EDA) - Preview
        # ====================================================================
        st.markdown("---")
        st.subheader("2.2 Exploratory Data Analysis (Preview)")

        if all_listings_list:
            listings_df = all_listings_list[0]  # Use first file for preview

            # Statistical Summary
            st.markdown("#### Statistical Summary")
            numeric_df = listings_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 0:
                with st.expander("📊 Descriptive Statistics", expanded=True):
                    st.dataframe(numeric_df.describe().round(2), use_container_width=True)

            # Price Distribution (if available)
            price_col = None
            for col in ['price', 'price_clean', 'listing_price']:
                if col in listings_df.columns:
                    price_col = col
                    break

            st.markdown("#### Key Distributions")
            dist_col1, dist_col2 = st.columns(2)

            with dist_col1:
                if price_col and listings_df[price_col].notna().sum() > 0:
                    # Clean price for visualization
                    price_data = listings_df[price_col].copy()
                    if price_data.dtype == 'object':
                        price_data = price_data.astype(str).replace(r'[\$,]', '', regex=True)
                        price_data = pd.to_numeric(price_data, errors='coerce')
                    price_valid = price_data.dropna()
                    if len(price_valid) > 0:
                        fig = px.histogram(x=price_valid, nbins=50, title="Price Distribution")
                        fig.update_layout(xaxis_title="Price", yaxis_title="Count", height=300)
                        st.plotly_chart(fig, use_container_width=True)

            with dist_col2:
                if 'room_type' in listings_df.columns:
                    room_counts = listings_df['room_type'].value_counts()
                    fig = px.pie(values=room_counts.values, names=room_counts.index, title="Room Type Distribution")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                elif 'property_type' in listings_df.columns:
                    prop_counts = listings_df['property_type'].value_counts().head(10)
                    fig = px.bar(x=prop_counts.index, y=prop_counts.values, title="Top 10 Property Types")
                    fig.update_layout(height=300, xaxis_title="Property Type", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

            # ====================================================================
            # OUTLIER ANALYSIS SECTION
            # ====================================================================
            st.markdown("#### 🔍 Outlier Analysis")
            st.markdown("*Identifying extreme values that may affect model accuracy*")

            if price_col and listings_df[price_col].notna().sum() > 0:
                # Clean price for analysis
                price_data = listings_df[price_col].copy()
                if price_data.dtype == 'object':
                    price_data = price_data.astype(str).replace(r'[\$,]', '', regex=True)
                    price_data = pd.to_numeric(price_data, errors='coerce')
                price_valid = price_data.dropna()

                if len(price_valid) > 0:
                    # Calculate IQR-based outliers
                    Q1 = price_valid.quantile(0.25)
                    Q3 = price_valid.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Identify outliers
                    outliers_mask = (price_valid < lower_bound) | (price_valid > upper_bound)
                    outliers = price_valid[outliers_mask]
                    n_outliers = len(outliers)
                    pct_outliers = (n_outliers / len(price_valid)) * 100

                    # Additional thresholds for context
                    extreme_high = price_valid[price_valid > Q3 + 3 * IQR]
                    extreme_low = price_valid[price_valid < Q1 - 3 * IQR]

                    # Display metrics
                    out_col1, out_col2, out_col3, out_col4 = st.columns(4)
                    with out_col1:
                        st.metric("Total Outliers", f"{n_outliers:,}", f"{pct_outliers:.1f}% of data")
                    with out_col2:
                        st.metric("IQR Lower Bound", f"${max(0, lower_bound):.0f}")
                    with out_col3:
                        st.metric("IQR Upper Bound", f"${upper_bound:.0f}")
                    with out_col4:
                        st.metric("Extreme High (>3×IQR)", f"{len(extreme_high):,}")

                    # === DOWNLOAD FULL OUTLIER DATA ===
                    st.markdown("---")
                    st.markdown("#### 📥 Download Full Outlier Data")
                    st.markdown("*Download complete outlier listings to analyze with Claude or other tools*")

                    # Get all outlier indices
                    all_outlier_indices = price_valid[outliers_mask].index

                    # Create comprehensive outlier dataframe
                    full_outlier_df = listings_df.loc[all_outlier_indices].copy()
                    full_outlier_df['outlier_price'] = price_valid.loc[all_outlier_indices]
                    full_outlier_df['price_vs_upper_bound'] = full_outlier_df['outlier_price'] - upper_bound
                    full_outlier_df['price_vs_lower_bound'] = full_outlier_df['outlier_price'] - lower_bound
                    full_outlier_df['is_high_outlier'] = full_outlier_df['outlier_price'] > upper_bound
                    full_outlier_df['is_low_outlier'] = full_outlier_df['outlier_price'] < lower_bound

                    # Sort by price (highest first)
                    full_outlier_df = full_outlier_df.sort_values('outlier_price', ascending=False)

                    # Convert to CSV
                    outlier_csv = full_outlier_df.to_csv(index=True)

                    dl_col1, dl_col2, dl_col3 = st.columns([2, 1, 1])
                    with dl_col1:
                        st.download_button(
                            label=f"⬇️ Download All {n_outliers:,} Outliers (CSV)",
                            data=outlier_csv,
                            file_name="outlier_listings_full.csv",
                            mime="text/csv",
                            help="Download complete data for all outlier listings with all columns",
                            use_container_width=True
                        )
                    with dl_col2:
                        st.metric("High Outliers", f"{(full_outlier_df['is_high_outlier']).sum():,}")
                    with dl_col3:
                        st.metric("Low Outliers", f"{(full_outlier_df['is_low_outlier']).sum():,}")

                    st.caption(f"📋 Contains {n_outliers:,} listings × {len(full_outlier_df.columns)} columns | Sorted by price (highest first)")
                    st.markdown("---")

                    # Box plot with outliers
                    with st.expander("📦 Price Box Plot (Outlier Visualization)", expanded=True):
                        box_col1, box_col2 = st.columns([2, 1])

                        with box_col1:
                            fig_box = go.Figure()
                            fig_box.add_trace(go.Box(
                                y=price_valid,
                                name="Price",
                                boxpoints='outliers',
                                fillcolor='lightblue',
                                line=dict(color='darkblue'),
                                marker=dict(
                                    outliercolor='red',
                                    color='blue',
                                    size=5
                                )
                            ))
                            fig_box.update_layout(
                                title="Price Distribution with Outliers",
                                yaxis_title="Price ($)",
                                height=400,
                                showlegend=False
                            )
                            # Add IQR bounds as horizontal lines
                            fig_box.add_hline(y=upper_bound, line_dash="dash", line_color="orange",
                                             annotation_text=f"Upper Bound: ${upper_bound:.0f}")
                            fig_box.add_hline(y=lower_bound, line_dash="dash", line_color="orange",
                                             annotation_text=f"Lower Bound: ${max(0, lower_bound):.0f}")
                            st.plotly_chart(fig_box, use_container_width=True)

                        with box_col2:
                            st.markdown("**📊 Price Statistics:**")
                            st.markdown(f"- **Min:** ${price_valid.min():.0f}")
                            st.markdown(f"- **Q1 (25%):** ${Q1:.0f}")
                            st.markdown(f"- **Median:** ${price_valid.median():.0f}")
                            st.markdown(f"- **Q3 (75%):** ${Q3:.0f}")
                            st.markdown(f"- **Max:** ${price_valid.max():.0f}")
                            st.markdown(f"- **Mean:** ${price_valid.mean():.0f}")
                            st.markdown(f"- **Std Dev:** ${price_valid.std():.0f}")
                            st.markdown("---")
                            st.markdown("**🔴 Outlier Thresholds:**")
                            st.markdown(f"- Below ${max(0, lower_bound):.0f}")
                            st.markdown(f"- Above ${upper_bound:.0f}")

                    # Outlier distribution by price range
                    with st.expander("📊 Outlier Distribution by Price Range"):
                        # Create price bins
                        bins = [0, 50, 100, 150, 200, 300, 500, 1000, float('inf')]
                        labels = ['$0-50', '$50-100', '$100-150', '$150-200', '$200-300', '$300-500', '$500-1000', '$1000+']

                        price_bins = pd.cut(price_valid, bins=bins, labels=labels, include_lowest=True)
                        outlier_bins = pd.cut(outliers, bins=bins, labels=labels, include_lowest=True)

                        # Count by bin
                        total_by_bin = price_bins.value_counts().sort_index()
                        outliers_by_bin = outlier_bins.value_counts().sort_index()

                        # Create comparison dataframe
                        bin_analysis = pd.DataFrame({
                            'Price Range': labels,
                            'Total Listings': [total_by_bin.get(l, 0) for l in labels],
                            'Outliers': [outliers_by_bin.get(l, 0) for l in labels]
                        })
                        bin_analysis['Pct of Range'] = (bin_analysis['Outliers'] / bin_analysis['Total Listings'] * 100).fillna(0)
                        bin_analysis['Pct of All Outliers'] = (bin_analysis['Outliers'] / n_outliers * 100).fillna(0) if n_outliers > 0 else 0

                        # Display table
                        st.dataframe(
                            bin_analysis.style.format({
                                'Pct of Range': '{:.1f}%',
                                'Pct of All Outliers': '{:.1f}%'
                            }).background_gradient(subset=['Outliers'], cmap='Reds'),
                            use_container_width=True
                        )

                        # Bar chart
                        fig_bins = go.Figure(data=[
                            go.Bar(name='Normal', x=labels, y=bin_analysis['Total Listings'] - bin_analysis['Outliers'], marker_color='lightblue'),
                            go.Bar(name='Outliers', x=labels, y=bin_analysis['Outliers'], marker_color='red')
                        ])
                        fig_bins.update_layout(
                            barmode='stack',
                            title="Listings by Price Range (Normal vs Outliers)",
                            xaxis_title="Price Range",
                            yaxis_title="Count",
                            height=350
                        )
                        st.plotly_chart(fig_bins, use_container_width=True)

                    # Scatter plot: Price vs Key Features with outliers highlighted
                    with st.expander("🔴 Outliers vs Key Features"):
                        # Find accommodation-related columns
                        feature_options = []
                        for col in ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'bathrooms_clean',
                                   'number_of_reviews', 'reviews_per_month', 'availability_365']:
                            if col in listings_df.columns:
                                feature_options.append(col)

                        if feature_options:
                            selected_feature = st.selectbox("Select feature to compare:", feature_options, key="outlier_feature")

                            # Create scatter data
                            scatter_df = pd.DataFrame({
                                'price': price_valid,
                                'feature': listings_df.loc[price_valid.index, selected_feature] if selected_feature in listings_df.columns else 0,
                                'is_outlier': outliers_mask
                            }).dropna()

                            if len(scatter_df) > 0:
                                fig_scatter = px.scatter(
                                    scatter_df,
                                    x='feature',
                                    y='price',
                                    color='is_outlier',
                                    color_discrete_map={True: 'red', False: 'blue'},
                                    opacity=0.6,
                                    title=f"Price vs {selected_feature} (Outliers in Red)",
                                    labels={'feature': selected_feature, 'price': 'Price ($)', 'is_outlier': 'Outlier'}
                                )
                                fig_scatter.update_layout(height=400)
                                st.plotly_chart(fig_scatter, use_container_width=True)

                                # Show outlier examples
                                st.markdown("**🔍 Sample Outlier Listings:**")
                                outlier_indices = price_valid[outliers_mask].sort_values(ascending=False).head(10).index

                                display_cols = ['id', 'name', price_col]
                                for col in ['accommodates', 'bedrooms', 'room_type', 'neighbourhood_cleansed']:
                                    if col in listings_df.columns:
                                        display_cols.append(col)

                                available_cols = [c for c in display_cols if c in listings_df.columns]
                                if available_cols and len(outlier_indices) > 0:
                                    outlier_samples = listings_df.loc[outlier_indices, available_cols].copy()
                                    st.dataframe(outlier_samples, use_container_width=True)

                                # Download full outlier data
                                st.markdown("---")
                                st.markdown("**📥 Download Full Outlier Data:**")
                                all_outlier_indices = price_valid[outliers_mask].index

                                # Get all columns for the full export
                                full_outlier_df = listings_df.loc[all_outlier_indices].copy()
                                full_outlier_df['outlier_price'] = price_valid.loc[all_outlier_indices]

                                # Convert to CSV
                                csv_buffer = full_outlier_df.to_csv(index=True)

                                st.download_button(
                                    label=f"⬇️ Download All {len(all_outlier_indices):,} Outliers (CSV)",
                                    data=csv_buffer,
                                    file_name="outlier_listings_full.csv",
                                    mime="text/csv",
                                    help="Download complete data for all outlier listings"
                                )
                                st.caption(f"Contains {len(all_outlier_indices):,} outlier listings with all available columns")
                        else:
                            st.info("No suitable features found for scatter plot comparison.")

                    # ============================================
                    # OUTLIERS BY ROOM TYPE ANALYSIS (Critical!)
                    # ============================================
                    with st.expander("🏠 Outliers by Room Type (Should You Remove Them?)", expanded=True):
                        st.markdown("""
                        **⚠️ Important:** Before removing outliers, check if they're actually valid data for specific room types!
                        High prices might be legitimate for "Entire home/apt" with many bedrooms.
                        """)

                        if 'room_type' in listings_df.columns:
                            # Create analysis dataframe
                            analysis_df = pd.DataFrame({
                                'price': price_valid,
                                'is_outlier': outliers_mask,
                                'room_type': listings_df.loc[price_valid.index, 'room_type']
                            })

                            # Add bedrooms if available
                            if 'bedrooms' in listings_df.columns:
                                analysis_df['bedrooms'] = listings_df.loc[price_valid.index, 'bedrooms']

                            # 1. Box plot by room type
                            st.markdown("##### 📊 Price Distribution by Room Type")
                            fig_room = px.box(analysis_df, x='room_type', y='price',
                                             color='room_type',
                                             title="Price Distribution by Room Type (Red line = Overall IQR Upper Bound)",
                                             points='outliers')
                            fig_room.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                                              annotation_text=f"Overall Upper Bound: ${upper_bound:.0f}")
                            fig_room.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig_room, use_container_width=True)

                            # 2. Statistics by room type
                            st.markdown("##### 📋 Statistics by Room Type")
                            room_stats = analysis_df.groupby('room_type').agg({
                                'price': ['count', 'mean', 'median', 'std', 'min', 'max',
                                         lambda x: x.quantile(0.75), lambda x: x.quantile(0.95)],
                                'is_outlier': ['sum', 'mean']
                            }).round(2)
                            room_stats.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q75', 'P95',
                                                 'Outliers', 'Outlier %']
                            room_stats['Outlier %'] = (room_stats['Outlier %'] * 100).round(1).astype(str) + '%'
                            room_stats['Mean'] = room_stats['Mean'].apply(lambda x: f"${x:.0f}")
                            room_stats['Median'] = room_stats['Median'].apply(lambda x: f"${x:.0f}")
                            room_stats['Q75'] = room_stats['Q75'].apply(lambda x: f"${x:.0f}")
                            room_stats['P95'] = room_stats['P95'].apply(lambda x: f"${x:.0f}")
                            room_stats['Max'] = room_stats['Max'].apply(lambda x: f"${x:.0f}")
                            st.dataframe(room_stats, use_container_width=True)

                            # 3. Where are the outliers coming from?
                            st.markdown("##### 🔍 Outlier Composition by Room Type")
                            outlier_composition = analysis_df[analysis_df['is_outlier']].groupby('room_type').size()
                            total_outliers_by_room = outlier_composition.sum()

                            if total_outliers_by_room > 0:
                                comp_col1, comp_col2 = st.columns(2)

                                with comp_col1:
                                    # Pie chart of outlier composition
                                    fig_pie = px.pie(values=outlier_composition.values,
                                                    names=outlier_composition.index,
                                                    title="Where Do Outliers Come From?")
                                    fig_pie.update_layout(height=300)
                                    st.plotly_chart(fig_pie, use_container_width=True)

                                with comp_col2:
                                    st.markdown("**Outlier Breakdown:**")
                                    for room, count in outlier_composition.items():
                                        pct = count / total_outliers_by_room * 100
                                        total_in_room = (analysis_df['room_type'] == room).sum()
                                        pct_of_room = count / total_in_room * 100 if total_in_room > 0 else 0
                                        st.write(f"- **{room}:** {count} outliers ({pct:.1f}% of all outliers, {pct_of_room:.1f}% of this room type)")

                            # 4. Outliers by room type AND bedrooms
                            if 'bedrooms' in analysis_df.columns:
                                st.markdown("##### 🛏️ Outliers by Room Type and Bedrooms")

                                # Heatmap of average price by room type and bedrooms
                                pivot_price = analysis_df.pivot_table(
                                    values='price', index='room_type', columns='bedrooms',
                                    aggfunc='median'
                                ).round(0)

                                fig_heat = px.imshow(pivot_price, text_auto=True,
                                                    title="Median Price by Room Type & Bedrooms",
                                                    color_continuous_scale='YlOrRd',
                                                    labels={'color': 'Price ($)'})
                                fig_heat.update_layout(height=300)
                                st.plotly_chart(fig_heat, use_container_width=True)

                                # Show that high prices correlate with bedrooms
                                st.markdown("**Key Insight:** Higher prices often correlate with more bedrooms, not random outliers!")

                            # 5. Decision recommendation
                            st.markdown("---")
                            st.markdown("##### 💡 Should You Remove These Outliers?")

                            # Calculate key metrics for decision
                            entire_home_outliers = outlier_composition.get('Entire home/apt', 0)
                            entire_home_pct = entire_home_outliers / total_outliers_by_room * 100 if total_outliers_by_room > 0 else 0

                            if entire_home_pct > 70:
                                st.warning(f"""
                                **⚠️ {entire_home_pct:.0f}% of outliers are "Entire home/apt"**

                                These are likely **legitimate high-value properties** (large homes, luxury rentals).
                                Removing them will hurt your model's ability to predict prices for premium listings.

                                **Recommendation:**
                                - ❌ Don't use aggressive outlier removal
                                - ✅ Consider **Price-Segmented Models** instead (trains separate models for Budget/Standard/Premium)
                                - ✅ Or use a **mild Hard Price Cap** (e.g., $500-$1000) to remove only extreme outliers
                                """)
                            elif entire_home_pct > 50:
                                st.info(f"""
                                **📊 {entire_home_pct:.0f}% of outliers are "Entire home/apt"**

                                Most outliers are larger properties with legitimately higher prices.

                                **Recommendation:**
                                - Use **moderate Hard Price Cap** (e.g., $500)
                                - Consider **Price-Segmented Models** for best results
                                """)
                            else:
                                st.success(f"""
                                **✅ Outliers are spread across room types**

                                Outliers appear to be true anomalies, not just different property categories.

                                **Recommendation:**
                                - Safe to use **Target Clipping** or **Hard Price Cap**
                                - IQR-based outlier removal is reasonable
                                """)

                            # Show what each option would do
                            st.markdown("**📈 Impact of Removal Options:**")
                            options_data = []

                            # Option 1: Remove all outliers
                            opt1_remaining = (~outliers_mask).sum()
                            opt1_entire_lost = analysis_df[(analysis_df['is_outlier']) & (analysis_df['room_type'] == 'Entire home/apt')].shape[0]

                            # Option 2: Only remove extreme outliers (> 3x IQR)
                            extreme_mask = price_valid > (Q3 + 3 * IQR)
                            opt2_remaining = (~extreme_mask).sum()
                            opt2_lost = extreme_mask.sum()

                            # Option 3: Room-type specific bounds
                            # Calculate IQR per room type
                            room_type_bounds = {}
                            for room in analysis_df['room_type'].unique():
                                room_prices = analysis_df[analysis_df['room_type'] == room]['price']
                                r_q1, r_q3 = room_prices.quantile(0.25), room_prices.quantile(0.75)
                                r_iqr = r_q3 - r_q1
                                room_type_bounds[room] = r_q3 + 1.5 * r_iqr

                            options_data = [
                                {
                                    'Option': 'Remove All IQR Outliers',
                                    'Removed': n_outliers,
                                    'Remaining': opt1_remaining,
                                    'Entire Homes Lost': opt1_entire_lost,
                                    'Risk': 'High - loses premium data'
                                },
                                {
                                    'Option': 'Remove Only Extreme (>3×IQR)',
                                    'Removed': opt2_lost,
                                    'Remaining': opt2_remaining,
                                    'Entire Homes Lost': analysis_df[(extreme_mask.reindex(analysis_df.index, fill_value=False)) & (analysis_df['room_type'] == 'Entire home/apt')].shape[0] if extreme_mask.sum() > 0 else 0,
                                    'Risk': 'Low - keeps most data'
                                },
                                {
                                    'Option': 'Hard Cap at $500',
                                    'Removed': (price_valid > 500).sum(),
                                    'Remaining': (price_valid <= 500).sum(),
                                    'Entire Homes Lost': analysis_df[(price_valid.reindex(analysis_df.index, fill_value=0) > 500) & (analysis_df['room_type'] == 'Entire home/apt')].shape[0],
                                    'Risk': 'Medium'
                                },
                                {
                                    'Option': 'Use Price-Segmented Models',
                                    'Removed': 0,
                                    'Remaining': len(price_valid),
                                    'Entire Homes Lost': 0,
                                    'Risk': 'None - keeps all data!'
                                }
                            ]
                            st.dataframe(pd.DataFrame(options_data), use_container_width=True)

                        else:
                            st.warning("Room type column not found. Cannot analyze outliers by category.")

                    # Recommendations based on outlier analysis
                    with st.expander("💡 Outlier Handling Recommendations", expanded=True):
                        st.markdown("**Based on your data analysis:**")

                        if pct_outliers > 10:
                            st.error(f"""
                            **⚠️ High Outlier Percentage ({pct_outliers:.1f}%)**

                            Your dataset has a significant number of outliers which will hurt model accuracy.

                            **Recommended actions:**
                            1. Use **Hard Price Cap** in Advanced Options (suggested: ${upper_bound:.0f})
                            2. Enable **Target Clipping** at 5% or higher
                            3. Consider filtering to core market: ${Q1:.0f} - ${Q3:.0f}
                            """)
                        elif pct_outliers > 5:
                            st.warning(f"""
                            **📊 Moderate Outliers ({pct_outliers:.1f}%)**

                            **Recommended actions:**
                            1. Use **Hard Price Cap** at ${upper_bound:.0f} to remove extreme high prices
                            2. Enable **Target Clipping** at 5%
                            """)
                        else:
                            st.success(f"""
                            **✅ Low Outlier Percentage ({pct_outliers:.1f}%)**

                            Your data has relatively few outliers. Standard preprocessing should work well.
                            """)

                        # Show impact estimate
                        if len(extreme_high) > 0:
                            st.markdown(f"""
                            **🎯 Potential Accuracy Impact:**
                            - Removing {len(extreme_high)} extreme high-price listings (>${Q3 + 3*IQR:.0f})
                              could improve MAPE by **2-5%**
                            - The {pct_outliers:.1f}% outliers contribute disproportionately to prediction errors
                            """)

                        # Quick stats on what filtering would do
                        st.markdown("**📈 Filtering Scenarios:**")
                        scenarios = [
                            (upper_bound, f"IQR Upper (${upper_bound:.0f})"),
                            (500, "Hard Cap $500"),
                            (300, "Hard Cap $300"),
                            (200, "Hard Cap $200")
                        ]

                        scenario_data = []
                        for cap, label in scenarios:
                            remaining = (price_valid <= cap).sum()
                            removed = len(price_valid) - remaining
                            new_mean = price_valid[price_valid <= cap].mean()
                            new_std = price_valid[price_valid <= cap].std()
                            scenario_data.append({
                                'Cap': label,
                                'Remaining': remaining,
                                'Removed': removed,
                                'Removed %': f"{(removed/len(price_valid)*100):.1f}%",
                                'New Mean': f"${new_mean:.0f}",
                                'New Std': f"${new_std:.0f}"
                            })

                        st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)
            else:
                st.info("Price column not found. Upload listings data with a 'price' column for outlier analysis.")

            # Correlation Analysis
            st.markdown("#### Correlation Analysis")
            if len(numeric_df.columns) > 1:
                # Select key numeric columns for correlation
                key_cols = [c for c in numeric_df.columns if any(x in c.lower() for x in
                           ['price', 'review', 'rating', 'bedroom', 'bathroom', 'accommodates', 'availability'])]
                if len(key_cols) > 1:
                    corr_df = numeric_df[key_cols].corr()
                    fig = px.imshow(corr_df, text_auto='.2f', aspect='auto',
                                   title="Correlation Heatmap (Key Variables)",
                                   color_continuous_scale='RdBu_r')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough key numeric columns for correlation analysis.")

            # Price Correlation Analysis (Critical for understanding R² potential)
            st.markdown("#### 💰 Price Correlation Analysis")
            st.markdown("*Features with higher correlation to price have more predictive power*")

            # Try to find and clean price column
            price_col_name = None
            for col in ['price', 'price_clean']:
                if col in listings_df.columns:
                    price_col_name = col
                    break

            if price_col_name:
                # Clean price if needed
                price_series = listings_df[price_col_name].copy()
                if price_series.dtype == 'object':
                    price_series = price_series.astype(str).replace(r'[\$,]', '', regex=True)
                    price_series = pd.to_numeric(price_series, errors='coerce')

                # Calculate correlations with price
                price_correlations = []
                for col in numeric_df.columns:
                    if col != price_col_name and numeric_df[col].notna().sum() > 100:
                        corr = price_series.corr(numeric_df[col])
                        if pd.notna(corr):
                            price_correlations.append({'Feature': col, 'Correlation': corr, 'Abs_Corr': abs(corr)})

                if price_correlations:
                    corr_df_price = pd.DataFrame(price_correlations).sort_values('Abs_Corr', ascending=False)

                    # Show top correlations
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown("**🟢 Top Positive Correlations:**")
                        top_pos = corr_df_price[corr_df_price['Correlation'] > 0].head(10)
                        for _, row in top_pos.iterrows():
                            bar_width = int(abs(row['Correlation']) * 100)
                            st.markdown(f"`{row['Feature'][:25]:25s}` {'█' * max(bar_width//5, 1)} **{row['Correlation']:.3f}**")

                    with col_neg:
                        st.markdown("**🔴 Top Negative Correlations:**")
                        top_neg = corr_df_price[corr_df_price['Correlation'] < 0].head(10)
                        for _, row in top_neg.iterrows():
                            bar_width = int(abs(row['Correlation']) * 100)
                            st.markdown(f"`{row['Feature'][:25]:25s}` {'█' * max(bar_width//5, 1)} **{row['Correlation']:.3f}**")

                    # R² Potential Analysis
                    max_corr = corr_df_price['Abs_Corr'].max()
                    avg_top10_corr = corr_df_price.head(10)['Abs_Corr'].mean()

                    st.markdown("---")
                    st.markdown("#### 📊 R² Score Potential Analysis")

                    r2_col1, r2_col2, r2_col3 = st.columns(3)
                    with r2_col1:
                        st.metric("Max Feature Correlation", f"{max_corr:.3f}")
                    with r2_col2:
                        st.metric("Avg Top-10 Correlation", f"{avg_top10_corr:.3f}")
                    with r2_col3:
                        # Theoretical max R² from single best feature
                        theoretical_r2 = max_corr ** 2
                        st.metric("Single Feature Max R²", f"{theoretical_r2:.1%}")

                    # Interpretation
                    if max_corr < 0.3:
                        st.error(f"""
                        **⚠️ Low Correlation Warning**

                        Your strongest feature correlation with price is only **{max_corr:.3f}**.

                        **Why R² is limited:**
                        - Airbnb prices depend heavily on **subjective factors** (photo quality, description appeal, exact micro-location)
                        - These factors aren't captured in structured data
                        - Expected R² range: **30-50%** is realistic for this data

                        **This is normal for Airbnb data** - even Airbnb's own models struggle with price prediction.
                        """)
                    elif max_corr < 0.5:
                        st.warning(f"""
                        **📊 Moderate Correlation**

                        Best feature correlation: **{max_corr:.3f}**

                        Expected R² range: **40-60%** with good feature engineering.
                        """)
                    else:
                        st.success(f"""
                        **✅ Good Correlation Potential**

                        Best feature correlation: **{max_corr:.3f}**

                        Expected R² range: **50-75%** achievable.
                        """)

            # Key Insights
            st.markdown("#### Key Insights")
            insights = []

            if price_col:
                price_data = listings_df[price_col].copy()
                if price_data.dtype == 'object':
                    price_data = price_data.astype(str).replace(r'[\$,]', '', regex=True)
                    price_data = pd.to_numeric(price_data, errors='coerce')
                if price_data.notna().sum() > 0:
                    insights.append(f"💰 **Price Range:** ${price_data.min():.0f} - ${price_data.max():.0f} (Median: ${price_data.median():.0f})")

            if 'room_type' in listings_df.columns:
                top_room = listings_df['room_type'].mode().iloc[0] if len(listings_df['room_type'].mode()) > 0 else "N/A"
                insights.append(f"🏠 **Most Common Room Type:** {top_room}")

            if 'neighbourhood_cleansed' in listings_df.columns:
                n_neighborhoods = listings_df['neighbourhood_cleansed'].nunique()
                insights.append(f"📍 **Neighborhoods Covered:** {n_neighborhoods}")

            if 'host_id' in listings_df.columns:
                n_hosts = listings_df['host_id'].nunique()
                avg_listings = len(listings_df) / n_hosts if n_hosts > 0 else 0
                insights.append(f"👤 **Unique Hosts:** {n_hosts:,} (Avg {avg_listings:.1f} listings/host)")

            for insight in insights:
                st.markdown(insight)

        # ====================================================================
        # 2.3 Feature Relevance to Business Problem
        # ====================================================================
        st.markdown("---")
        st.subheader("2.3 Feature Relevance to Business Problem")

        st.markdown("""
        **Business Problem:** Predict Airbnb listing prices to help hosts optimize pricing and guests find value.

        #### Feature Categories and Their Relevance:
        """)

        feature_relevance = {
            "🏠 Property Features": {
                "columns": ["accommodates", "bedrooms", "bathrooms", "beds", "property_type", "room_type"],
                "relevance": "Directly impact pricing capacity and guest expectations. Larger properties with more amenities typically command higher prices."
            },
            "📍 Location Features": {
                "columns": ["neighbourhood_cleansed", "latitude", "longitude"],
                "relevance": "Location is a primary price driver. Central/popular neighborhoods have premium pricing."
            },
            "⭐ Review & Rating Features": {
                "columns": ["review_scores_rating", "number_of_reviews", "reviews_per_month"],
                "relevance": "Social proof affects booking rates and pricing power. Higher-rated listings can charge premium prices."
            },
            "👤 Host Features": {
                "columns": ["host_is_superhost", "host_response_rate", "host_listings_count"],
                "relevance": "Host quality signals trustworthiness. Superhosts and responsive hosts may command higher prices."
            },
            "📅 Availability Features": {
                "columns": ["availability_30", "availability_365", "minimum_nights"],
                "relevance": "Booking flexibility and scarcity affect demand-based pricing strategies."
            },
            "📝 Text Features": {
                "columns": ["name", "description", "amenities"],
                "relevance": "Marketing quality and amenity offerings differentiate listings and justify price premiums."
            }
        }

        if st.session_state.listings is not None:
            listings_df = st.session_state.listings
            for category, info in feature_relevance.items():
                with st.expander(category):
                    st.markdown(f"**Relevance:** {info['relevance']}")
                    available = [c for c in info['columns'] if c in listings_df.columns]
                    missing = [c for c in info['columns'] if c not in listings_df.columns]
                    st.markdown(f"**Available in dataset:** {', '.join(available) if available else 'None'}")
                    if missing:
                        st.markdown(f"**Missing:** {', '.join(missing)}")

        # ====================================================================
        # 2.4 DU Output: Feasible Model Classes
        # ====================================================================
        st.markdown("---")
        st.subheader("2.4 Feasible Model Classes")

        st.markdown("""
        Based on the Airbnb dataset structure (tabular data with mixed numeric/categorical features),
        the following ML model families are suitable for price prediction:
        """)

        model_families = {
            "📈 Linear Models": {
                "models": ["Linear Regression", "Ridge", "Lasso", "ElasticNet"],
                "pros": "Interpretable, fast training, good baseline",
                "cons": "May underfit complex non-linear relationships",
                "suitable": True
            },
            "🌳 Tree-Based Models": {
                "models": ["Decision Tree", "Random Forest", "Extra Trees"],
                "pros": "Handle non-linearity, feature importance, no scaling needed",
                "cons": "Can overfit, less interpretable than linear",
                "suitable": True
            },
            "🚀 Gradient Boosting": {
                "models": ["XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"],
                "pros": "State-of-the-art for tabular data, handles mixed types well",
                "cons": "Requires tuning, longer training time",
                "suitable": True
            },
            "🔮 Support Vector Machines": {
                "models": ["SVR (RBF kernel)"],
                "pros": "Effective in high-dimensional spaces",
                "cons": "Slow on large datasets, requires scaling",
                "suitable": True
            },
            "🧠 Neural Networks": {
                "models": ["MLP Regressor"],
                "pros": "Can learn complex patterns",
                "cons": "Needs more data, harder to tune, less interpretable",
                "suitable": True
            },
            "🏘️ Instance-Based": {
                "models": ["K-Nearest Neighbors"],
                "pros": "Simple, no training phase",
                "cons": "Slow predictions on large data, sensitive to scale",
                "suitable": True
            }
        }

        for family, info in model_families.items():
            with st.expander(f"{family} {'✅' if info['suitable'] else '❌'}"):
                st.markdown(f"**Models:** {', '.join(info['models'])}")
                st.markdown(f"**Pros:** {info['pros']}")
                st.markdown(f"**Cons:** {info['cons']}")

        st.info("""
        **Recommendation:** For Airbnb price prediction, **Gradient Boosting models** (XGBoost, LightGBM, CatBoost)
        typically perform best due to their ability to handle mixed feature types and capture non-linear relationships.
        Start with Random Forest as a strong baseline, then optimize with gradient boosting methods.
        """)

# ============================================================================
# TAB 2: DATA CLEANING
# ============================================================================
with tabs[1]:
    st.header("2️⃣ Data Cleaning & Quality Analysis")

    # Check if data was uploaded
    all_listings_uploaded = st.session_state.get('all_listings') or []
    if not all_listings_uploaded:
        st.warning("⚠️ Upload data first!")
    else:
        # ========================================
        # STEP 1: COMBINE DATASETS
        # ========================================
        st.subheader("📦 Step 1: Combine Datasets")

        dataset_sources = st.session_state.get('dataset_sources') or []
        if len(dataset_sources) > 1:
            st.info(f"📊 **Datasets to combine:** {', '.join(dataset_sources)}")

        # Show what will be combined
        all_listings_list = st.session_state.get('all_listings') or []
        all_calendar_list = st.session_state.get('all_calendar') or []
        all_reviews_list = st.session_state.get('all_reviews') or []
        all_neigh_list = st.session_state.get('all_neighbourhoods') or []

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_listings = len(all_listings_list)
            total_listings = sum(len(df) for df in all_listings_list)
            st.metric("🏠 Listings", f"{total_listings:,}", f"{n_listings} file(s)")
        with col2:
            n_cal = len(all_calendar_list)
            total_cal = sum(len(df) for df in all_calendar_list)
            st.metric("📅 Calendar", f"{total_cal:,}", f"{n_cal} file(s)")
        with col3:
            n_rev = len(all_reviews_list)
            total_rev = sum(len(df) for df in all_reviews_list)
            st.metric("⭐ Reviews", f"{total_rev:,}", f"{n_rev} file(s)")
        with col4:
            n_neigh = len(all_neigh_list)
            total_neigh = sum(len(df) for df in all_neigh_list)
            st.metric("📍 Neighbourhoods", f"{total_neigh:,}", f"{n_neigh} file(s)")

        # Combine button
        data_combined = st.session_state.get('data_combined', False)

        if not data_combined:
            if st.button("🔗 Combine Datasets", type="primary", use_container_width=True):
                with st.spinner("Combining datasets..."):
                    # Combine listings
                    if st.session_state.get('all_listings'):
                        combined_listings = pd.concat(st.session_state.all_listings, ignore_index=True)
                        st.session_state.listings = combined_listings
                        st.write(f"✅ Combined {len(st.session_state.all_listings)} listings file(s) → {len(combined_listings):,} rows")

                    # Combine calendar
                    if st.session_state.get('all_calendar'):
                        combined_calendar = pd.concat(st.session_state.all_calendar, ignore_index=True)
                        st.session_state.calendar = combined_calendar
                        st.write(f"✅ Combined {len(st.session_state.all_calendar)} calendar file(s) → {len(combined_calendar):,} rows")

                    # Combine reviews
                    if st.session_state.get('all_reviews'):
                        combined_reviews = pd.concat(st.session_state.all_reviews, ignore_index=True)
                        st.session_state.reviews = combined_reviews
                        st.write(f"✅ Combined {len(st.session_state.all_reviews)} reviews file(s) → {len(combined_reviews):,} rows")

                    # Combine neighbourhoods (with dedup)
                    if st.session_state.get('all_neighbourhoods'):
                        combined_neighbourhoods = pd.concat(st.session_state.all_neighbourhoods, ignore_index=True)
                        if 'neighbourhood' in combined_neighbourhoods.columns:
                            before_dedup = len(combined_neighbourhoods)
                            combined_neighbourhoods = combined_neighbourhoods.drop_duplicates(subset=['neighbourhood'], keep='last')
                            st.write(f"✅ Combined {len(st.session_state.all_neighbourhoods)} neighbourhood file(s) → {len(combined_neighbourhoods):,} rows (removed {before_dedup - len(combined_neighbourhoods)} duplicates)")
                        st.session_state.neighbourhoods = combined_neighbourhoods

                    st.session_state.data_combined = True
                    st.success("✅ Datasets combined successfully! Scroll down to configure and start cleaning.")
        else:
            st.success(f"✅ **Datasets Combined Successfully!**")

            # Show combined data summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏠 Listings", f"{len(st.session_state.listings):,}", "Combined")
            with col2:
                if st.session_state.calendar is not None:
                    st.metric("📅 Calendar", f"{len(st.session_state.calendar):,}", "Combined")
                else:
                    st.metric("📅 Calendar", "N/A")
            with col3:
                if st.session_state.reviews is not None:
                    st.metric("⭐ Reviews", f"{len(st.session_state.reviews):,}", "Combined")
                else:
                    st.metric("⭐ Reviews", "N/A")
            with col4:
                if st.session_state.neighbourhoods is not None:
                    st.metric("📍 Neighbourhoods", f"{len(st.session_state.neighbourhoods):,}", "Combined")
                else:
                    st.metric("📍 Neighbourhoods", "N/A")

            st.info("👇 **Next:** Configure cleaning options below and click 'Start Cleaning Process'")

        st.markdown("---")

        # Only show cleaning options if data is combined
        if not st.session_state.get('data_combined', False):
            st.warning("⚠️ Click 'Combine Datasets' above before proceeding with cleaning.")
        else:
            # Cleaning options
            st.subheader("⚙️ Step 2: Cleaning Configuration")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Basic Cleaning**")
                opt_duplicates = st.checkbox("🔄 Remove duplicate listings", value=True)
                opt_prices = st.checkbox("💰 Clean price formats ($, commas)", value=True)
                opt_pct = st.checkbox("📊 Clean percentage formats (%)", value=True)
                opt_missing = st.checkbox("📝 Handle missing values", value=True)

            with col2:
                st.markdown("**Advanced Cleaning**")
                opt_coords = st.checkbox("🌍 Validate coordinates", value=True)
                opt_dates = st.checkbox("📅 Validate dates", value=True)
                opt_outliers = st.checkbox("📈 Remove price outliers", value=True)
                opt_text = st.checkbox("📝 Normalize text fields", value=True)

            # Outlier method
            if opt_outliers:
                outlier_method = st.selectbox(
                    "Outlier Detection Method:",
                    ["Percentile (3-97%)", "IQR (1.5x)", "Z-Score (3σ)"],
                    help="Percentile: removes top/bottom X%. IQR: removes values outside Q1-1.5*IQR to Q3+1.5*IQR. Z-Score: removes values >3 standard deviations from mean."
                )
            else:
                outlier_method = None

            st.markdown("---")

            if st.button("🧹 Start Cleaning Process", type="primary", use_container_width=True):

                # Initialize logger
                logger = CleaningLogger()
                progress = st.progress(0)
                status_container = st.container()

                with status_container:
                    st.subheader("📋 Cleaning Progress")

                # ========================================
                # LISTINGS CLEANING
                # ========================================
                st.markdown("### 🏠 Cleaning Listings Data")
                listings = st.session_state.listings.copy()
                original_listings = len(listings)

                logger.log(f"Starting listings cleaning: {original_listings:,} rows")

                # Data quality analysis
                progress.progress(5)
                with st.expander("📊 Initial Data Quality Analysis", expanded=True):
                    quality = analyze_data_quality(listings, logger)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", f"{quality['total_rows']:,}")
                    with col2:
                        st.metric("Columns", quality['total_cols'])
                    with col3:
                        st.metric("Missing %", f"{quality['missing_pct']:.1f}%")
                    with col4:
                        st.metric("Numeric Cols", quality['numeric_cols'])

                # 1. Remove duplicates (based on content, not ID since IDs may not be unique across datasets)
                progress.progress(10)
                if opt_duplicates:
                    with st.expander("🔄 Duplicate Removal", expanded=True):
                        # Use multiple columns to identify true duplicates (not ID since it's not unique across datasets)
                        dup_cols = []
                        for col in ['name', 'latitude', 'longitude', 'host_id']:
                            if col in listings.columns:
                                dup_cols.append(col)

                        if len(dup_cols) >= 2:
                            listings, dup_removed = remove_duplicates_with_log(listings, dup_cols, logger)
                            st.write(f"Using columns for duplicate detection: {', '.join(dup_cols)}")

                            stats = logger.get_stats().get("Duplicate Removal", {})
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original", f"{stats.get('Original rows', 0):,}")
                            with col2:
                                st.metric("Removed", f"{stats.get('Rows removed', 0):,}")
                            with col3:
                                st.metric("Remaining", f"{stats.get('Final rows', 0):,}")
                        else:
                            st.info("ℹ️ Skipping duplicate removal - not enough identifying columns found")

                # 2. Clean prices
                progress.progress(20)
                if opt_prices and 'price' in listings.columns:
                    with st.expander("💰 Price Cleaning", expanded=True):
                        st.write("**Before cleaning (samples):**")
                        st.code(listings['price'].head(5).tolist())

                        listings['price_clean'] = clean_price_with_log(listings['price'], logger)

                        st.write("**After cleaning (samples):**")
                        st.code(listings['price_clean'].head(5).tolist())

                        stats = logger.get_stats().get("Price Cleaning", {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Valid Prices", f"{stats.get('Valid after cleaning', 0):,}")
                        with col2:
                            st.metric("Invalid (NaN)", f"{stats.get('Invalid (NaN)', 0):,}")
                        with col3:
                            st.metric("Had $ Symbol", f"{stats.get('Had $ symbol', 0):,}")

                        # Price distribution
                        valid_prices = listings['price_clean'].dropna()
                        if len(valid_prices) > 0:
                            st.write("**Price Distribution:**")
                            st.write(f"  • Min: ${valid_prices.min():.2f}")
                            st.write(f"  • Max: ${valid_prices.max():.2f}")
                            st.write(f"  • Mean: ${valid_prices.mean():.2f}")
                            st.write(f"  • Median: ${valid_prices.median():.2f}")

                # 3. Clean percentages
                progress.progress(30)
                if opt_pct:
                    pct_cols = ['host_response_rate', 'host_acceptance_rate']
                    pct_cols_exist = [c for c in pct_cols if c in listings.columns]

                    if pct_cols_exist:
                        with st.expander("📊 Percentage Cleaning", expanded=True):
                            for col in pct_cols_exist:
                                st.write(f"**Column: {col}**")
                                st.write(f"  Before: {listings[col].head(3).tolist()}")
                                listings[col] = clean_percentage_with_log(listings[col], col, logger)
                                st.write(f"  After: {listings[col].head(3).tolist()}")
                                st.write("")

                # 4. Validate coordinates
                progress.progress(40)
                if opt_coords and 'latitude' in listings.columns and 'longitude' in listings.columns:
                    with st.expander("🌍 Coordinate Validation", expanded=True):
                        valid_coords = validate_coordinates_with_log(listings, logger)
                        listings['valid_coordinates'] = valid_coords

                        stats = logger.get_stats().get("Coordinate Validation", {})

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Validation Results:**")
                            for key, value in stats.items():
                                st.write(f"  • {key}: {value}")
                        with col2:
                            # Mini map
                            if len(listings) > 0:
                                sample = listings.head(500)
                                fig = px.scatter_mapbox(
                                    sample, lat='latitude', lon='longitude',
                                    color='valid_coordinates',
                                    zoom=8, height=300,
                                    title="Coordinate Distribution"
                                )
                                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
                                st.plotly_chart(fig, use_container_width=True)

                # 5. Handle missing values
                progress.progress(50)
                if opt_missing:
                    with st.expander("📝 Missing Value Handling", expanded=True):
                        numeric_cols = ['bathrooms', 'bedrooms', 'beds', 'accommodates',
                                        'review_scores_rating', 'review_scores_cleanliness',
                                        'review_scores_checkin', 'review_scores_communication',
                                        'review_scores_location', 'review_scores_value',
                                        'host_response_rate', 'host_acceptance_rate']

                        existing_cols = [c for c in numeric_cols if c in listings.columns]
                        listings, missing_report, total_filled = handle_missing_with_log(listings, existing_cols, logger)

                        if missing_report:
                            st.write("**Missing Values Filled:**")
                            df_missing = pd.DataFrame(missing_report)
                            st.dataframe(df_missing, use_container_width=True)
                            st.metric("Total Values Filled", f"{total_filled:,}")
                        else:
                            st.success("No missing values found in numeric columns!")

                        # Fill defaults
                        defaults = {
                            'host_listings_count': 1,
                            'reviews_per_month': 0,
                            'minimum_nights': 1,
                            'number_of_reviews': 0
                        }
                        for col, default in defaults.items():
                            if col in listings.columns:
                                filled = listings[col].isna().sum()
                                if filled > 0:
                                    listings[col] = listings[col].fillna(default)
                                    st.write(f"  • {col}: filled {filled:,} with default={default}")

                # 6. Remove outliers
                progress.progress(60)
                if opt_outliers and 'price_clean' in listings.columns:
                    with st.expander("📈 Outlier Removal", expanded=True):
                        before_outliers = len(listings)

                        # Determine method
                        if outlier_method == "Percentile (3-97%)":
                            outliers, lower, upper = detect_outliers_with_log(
                                listings['price_clean'].dropna(), logger,
                                method='percentile', lower=0.03, upper=0.97
                            )
                            Q1, Q3 = listings['price_clean'].quantile(0.03), listings['price_clean'].quantile(0.97)
                            listings = listings[(listings['price_clean'] >= Q1) & (listings['price_clean'] <= Q3)]
                        elif outlier_method == "IQR (1.5x)":
                            outliers, lower, upper = detect_outliers_with_log(
                                listings['price_clean'].dropna(), logger,
                                method='iqr', multiplier=1.5
                            )
                            listings = listings[(listings['price_clean'] >= lower) & (listings['price_clean'] <= upper)]
                        else:  # Z-Score
                            outliers, lower, upper = detect_outliers_with_log(
                                listings['price_clean'].dropna(), logger,
                                method='zscore', threshold=3
                            )
                            listings = listings[(listings['price_clean'] >= lower) & (listings['price_clean'] <= upper)]

                        # Remove zero/negative
                        listings = listings[listings['price_clean'] > 0]

                        outliers_removed = before_outliers - len(listings)

                        # Store outlier bounds in session state for use in calendar cleaning
                        st.session_state.outlier_bounds = {
                            'lower': lower,
                            'upper': upper,
                            'method': outlier_method
                        }

                        stats = logger.get_stats().get("Outlier Detection", {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Method", stats.get('Method', 'N/A'))
                        with col2:
                            st.metric("Removed", f"{outliers_removed:,}")
                        with col3:
                            st.metric("Remaining", f"{len(listings):,}")

                        st.write(
                            f"**Valid price range:** {stats.get('Lower bound', 'N/A')} to {stats.get('Upper bound', 'N/A')}")

                        # Price histogram
                        fig = px.histogram(listings, x='price_clean', nbins=50, title="Price Distribution After Cleaning")
                        fig.add_vline(x=lower, line_dash="dash", line_color="red", annotation_text="Lower")
                        fig.add_vline(x=upper, line_dash="dash", line_color="red", annotation_text="Upper")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # No outlier removal - clear any previous bounds
                    st.session_state.outlier_bounds = None

                st.session_state.listings = listings

                # ========================================
                # CALENDAR CLEANING
                # ========================================
                progress.progress(70)
                if st.session_state.calendar is not None:
                    st.markdown("### 📅 Cleaning Calendar Data")
                    calendar = st.session_state.calendar.copy()
                    original_cal = len(calendar)

                    with st.expander("📅 Calendar Cleaning Details", expanded=True):
                        logger.log(f"Starting calendar cleaning: {original_cal:,} rows")

                        # Clean prices
                        if 'price' in calendar.columns:
                            calendar['price_cal'] = clean_price_with_log(calendar['price'], logger)

                            # Remove invalid prices (> 0)
                            before = len(calendar)
                            calendar = calendar[calendar['price_cal'].notna()]
                            calendar = calendar[calendar['price_cal'] > 0]

                            # Apply same outlier bounds as listings (if available)
                            outlier_bounds = st.session_state.get('outlier_bounds')
                            if outlier_bounds is not None:
                                cal_lower = outlier_bounds['lower']
                                cal_upper = outlier_bounds['upper']
                                cal_method = outlier_bounds['method']
                                before_outlier = len(calendar)
                                calendar = calendar[(calendar['price_cal'] >= cal_lower) & (calendar['price_cal'] <= cal_upper)]
                                outlier_removed = before_outlier - len(calendar)
                                st.write(f"✅ Applied {cal_method} outlier bounds (${cal_lower:.2f} - ${cal_upper:.2f})")
                                st.write(f"   Removed {outlier_removed:,} calendar outliers")
                            else:
                                # Fallback to basic $5000 cap if no outlier bounds set
                                calendar = calendar[calendar['price_cal'] < 5000]
                                st.write("ℹ️ Using default $5000 price cap (no outlier method selected)")

                            removed = before - len(calendar)
                            st.write(f"✅ Cleaned prices: removed {removed:,} invalid/outlier rows total")

                        # Convert availability
                        if 'available' in calendar.columns:
                            calendar['is_available'] = (calendar['available'] == 't').astype(int)
                            avail_rate = calendar['is_available'].mean()
                            st.write(f"✅ Availability converted: {avail_rate:.1%} available")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original", f"{original_cal:,}")
                        with col2:
                            st.metric("After Cleaning", f"{len(calendar):,}")
                        with col3:
                            st.metric("Removed", f"{original_cal - len(calendar):,}")

                    st.session_state.calendar = calendar

                # ========================================
                # REVIEWS CLEANING
                # ========================================
                progress.progress(85)
                if st.session_state.reviews is not None:
                    st.markdown("### ⭐ Cleaning Reviews Data")
                    reviews = st.session_state.reviews.copy()
                    original_rev = len(reviews)

                    with st.expander("⭐ Reviews Cleaning Details", expanded=True):
                        # Remove duplicates
                        if 'id' in reviews.columns:
                            before = len(reviews)
                            reviews = reviews.drop_duplicates(subset=['id'])
                            dup_removed = before - len(reviews)
                            st.write(f"✅ Removed {dup_removed:,} duplicate reviews")

                        # Clean comments
                        if 'comments' in reviews.columns:
                            reviews['comments'] = reviews['comments'].fillna('')
                            reviews['comment_length'] = reviews['comments'].str.len()

                            empty = (reviews['comment_length'] == 0).sum()
                            avg_len = reviews['comment_length'].mean()

                            st.write(f"✅ Comments analyzed: {empty:,} empty, avg length {avg_len:.0f} chars")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original", f"{original_rev:,}")
                        with col2:
                            st.metric("After Cleaning", f"{len(reviews):,}")

                    st.session_state.reviews = reviews

                # ========================================
                # FINAL SUMMARY
                # ========================================
                progress.progress(100)

                st.markdown("---")
                st.subheader("📋 Cleaning Summary")

                # Save logger
                st.session_state.cleaning_logger = logger

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    reduction = original_listings - len(st.session_state.listings)
                    st.metric("🏠 Listings", f"{len(st.session_state.listings):,}", f"-{reduction:,}")
                with col2:
                    if st.session_state.calendar is not None:
                        st.metric("📅 Calendar", f"{len(st.session_state.calendar):,}")
                with col3:
                    if st.session_state.reviews is not None:
                        st.metric("⭐ Reviews", f"{len(st.session_state.reviews):,}")
                with col4:
                    st.metric("✅ Operations", len(logger.get_logs()))

                # Full log
                with st.expander("📜 Complete Cleaning Log"):
                    logs = logger.get_logs()
                    for log in logs:
                        icon = "✅" if log['level'] == 'success' else "ℹ️"
                        st.write(f"`{log['time']}` {icon} {log['message']}")

                # All stats
                with st.expander("📊 All Statistics"):
                    all_stats = logger.get_stats()
                    for category, stats in all_stats.items():
                        st.write(f"**{category}**")
                        for key, value in stats.items():
                            st.write(f"  • {key}: {value}")
                        st.write("")

                st.success("✅ Data cleaning complete! Proceed to Processing tab.")

# ============================================================================
# TAB 3: PROCESS
# ============================================================================
with tabs[2]:
    st.header("3️⃣ Data Processing & Aggregation")

    if st.session_state.listings is None:
        st.warning("⚠️ Upload and clean data first!")
    else:
        st.markdown("""
        This step aggregates data from all sources:
        - **Calendar** → Price statistics (mean, std, min, max) per listing
        - **Reviews** → Sentiment scores per listing  
        - **Neighbourhoods** → Geographic features
        - **Merge** → Combine into single dataset
        """)

        if st.button("⚙️ Process & Aggregate", type="primary", use_container_width=True):
            progress = st.progress(0)

            listings = st.session_state.listings.copy()
            calendar = st.session_state.calendar.copy() if st.session_state.calendar is not None else None
            reviews = st.session_state.reviews.copy() if st.session_state.reviews is not None else None
            neighbourhoods = st.session_state.neighbourhoods.copy() if st.session_state.neighbourhoods is not None else None

            # NEIGHBOURHOODS
            progress.progress(10)
            if neighbourhoods is not None:
                with st.expander("📍 Neighbourhood Processing", expanded=True):
                    def extract_polygon_stats(geom_str):
                        try:
                            coords = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', str(geom_str))
                            if not coords: return None, None, None, None
                            lons = [float(c[0]) for c in coords]
                            lats = [float(c[1]) for c in coords]
                            return np.mean(lons), np.mean(lats), (max(lons) - min(lons)) * (max(lats) - min(lats)), \
                                sum(np.sqrt((lons[i] - lons[i - 1]) ** 2 + (lats[i] - lats[i - 1]) ** 2) for i in
                                    range(1, len(lons)))
                        except:
                            return None, None, None, None


                    neighbourhoods['neigh_centroid_lon'], neighbourhoods['neigh_centroid_lat'], \
                        neighbourhoods['neigh_area'], neighbourhoods['neigh_perimeter'] = \
                        zip(*neighbourhoods['geometry'].apply(extract_polygon_stats))

                    le_ng = LabelEncoder()
                    neighbourhoods['neigh_group_enc'] = le_ng.fit_transform(
                        neighbourhoods['neighbourhood_group'].fillna('Unknown'))

                    neigh_features = neighbourhoods[['neighbourhood', 'neigh_centroid_lon', 'neigh_centroid_lat',
                                                     'neigh_area', 'neigh_perimeter', 'neigh_group_enc']].copy()

                    lpc = listings.groupby('neighbourhood_cleansed').size().reset_index(name='neigh_listing_count')
                    neigh_features = neigh_features.merge(lpc, left_on='neighbourhood',
                                                          right_on='neighbourhood_cleansed', how='left')
                    neigh_features['neigh_listing_count'] = neigh_features['neigh_listing_count'].fillna(0)
                    neigh_features = neigh_features.drop(columns=['neighbourhood_cleansed'], errors='ignore')

                    ns = listings.groupby('neighbourhood_cleansed').agg({
                        'review_scores_rating': 'mean', 'accommodates': 'mean',
                        'bedrooms': 'mean', 'number_of_reviews': 'mean'
                    }).reset_index()
                    ns.columns = ['neighbourhood', 'neigh_avg_rating', 'neigh_avg_accommodates',
                                  'neigh_avg_bedrooms', 'neigh_avg_reviews']
                    neigh_features = neigh_features.merge(ns, on='neighbourhood', how='left')

                    st.write(f"✅ Processed {len(neigh_features)} neighbourhoods")
                    st.write(f"   Features: centroid, area, perimeter, listing_count, avg_rating, etc.")
            else:
                neigh_features = None

            # CALENDAR
            progress.progress(35)
            if calendar is not None:
                with st.expander("📅 Calendar Aggregation", expanded=True):
                    if 'price_cal' not in calendar.columns:
                        calendar['price_cal'] = pd.to_numeric(calendar['price'].replace(r'[\$,]', '', regex=True),
                                                              errors='coerce')
                        # Apply same outlier bounds as listings (if available)
                        outlier_bounds = st.session_state.get('outlier_bounds')
                        if outlier_bounds is not None:
                            cal_lower = outlier_bounds['lower']
                            cal_upper = outlier_bounds['upper']
                            calendar = calendar[(calendar['price_cal'] > 0) &
                                               (calendar['price_cal'] >= cal_lower) &
                                               (calendar['price_cal'] <= cal_upper)]
                        else:
                            # Fallback to basic $5000 cap if no outlier bounds set
                            calendar = calendar[(calendar['price_cal'] > 0) & (calendar['price_cal'] < 5000)]

                    if 'is_available' not in calendar.columns:
                        calendar['is_available'] = (calendar['available'] == 't').astype(int)

                    st.write(f"Aggregating {len(calendar):,} calendar rows...")

                    cal_agg = calendar.groupby('listing_id').agg({
                        'price_cal': ['mean', 'std', 'min', 'max', 'median',
                                      lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
                        'is_available': 'mean'
                    }).reset_index()
                    cal_agg.columns = ['listing_id', 'cal_price_mean', 'cal_price_std', 'cal_price_min',
                                       'cal_price_max', 'cal_price_median', 'cal_price_q25', 'cal_price_q75',
                                       'cal_avail_rate']

                    cal_agg['price_range'] = cal_agg['cal_price_max'] - cal_agg['cal_price_min']
                    cal_agg['price_iqr'] = cal_agg['cal_price_q75'] - cal_agg['cal_price_q25']
                    cal_agg['price_volatility'] = cal_agg['cal_price_std'] / cal_agg['cal_price_mean'].replace(0, 1)
                    cal_agg['dynamic_pricing'] = (cal_agg['price_range'] > 10).astype(int)
                    cal_agg['high_demand'] = (cal_agg['cal_avail_rate'] < 0.3).astype(int)

                    st.write(f"✅ Created {len(cal_agg):,} listing aggregations with 13 features")
                    st.write(f"   Avg price: ${cal_agg['cal_price_mean'].mean():.2f}")
                    st.write(f"   Avg availability: {cal_agg['cal_avail_rate'].mean():.1%}")

                    # Monthly price analysis
                    st.markdown("---")
                    st.markdown("##### 📅 Monthly Price Analysis")

                    # Parse dates and extract month
                    calendar['date'] = pd.to_datetime(calendar['date'], errors='coerce')
                    calendar['month'] = calendar['date'].dt.month
                    calendar['month_name'] = calendar['date'].dt.strftime('%B')
                    calendar['year'] = calendar['date'].dt.year

                    # DIAGNOSTIC: Show calendar data info
                    with st.expander("🔍 Calendar Data Diagnostics", expanded=False):
                        st.write(f"**Total calendar rows:** {len(calendar):,}")
                        st.write(f"**Unique listing_ids:** {calendar['listing_id'].nunique():,}")
                        st.write(f"**Unique dates:** {calendar['date'].nunique():,}")
                        st.write(f"**Date range:** {calendar['date'].min()} to {calendar['date'].max()}")
                        st.write(f"**Unique months in data:** {sorted(calendar['month'].dropna().unique().tolist())}")
                        st.write(f"**Unique prices:** {calendar['price_cal'].nunique():,}")
                        st.write(f"**Price range:** ${calendar['price_cal'].min():.2f} to ${calendar['price_cal'].max():.2f}")

                        # Check if prices vary by listing or are static
                        price_var_per_listing = calendar.groupby('listing_id')['price_cal'].std().mean()
                        st.write(f"**Avg price std per listing:** ${price_var_per_listing:.2f}")

                        if price_var_per_listing < 1:
                            st.warning("⚠️ Listings have very low price variation - hosts may not be using dynamic pricing!")

                        # Sample of calendar data
                        st.write("**Sample calendar rows:**")
                        sample_cols = ['listing_id', 'date', 'price_cal', 'month', 'is_available']
                        sample_cols = [c for c in sample_cols if c in calendar.columns]
                        st.dataframe(calendar[sample_cols].head(10), use_container_width=True)

                    # Calculate monthly statistics
                    monthly_stats = calendar.groupby('month').agg({
                        'price_cal': ['mean', 'median', 'std', 'count'],
                        'is_available': 'mean'
                    }).reset_index()
                    monthly_stats.columns = ['month', 'avg_price', 'median_price', 'price_std', 'data_points', 'availability']

                    # Calculate price index (relative to overall mean)
                    overall_mean = calendar['price_cal'].mean()
                    monthly_stats['price_index_raw'] = monthly_stats['avg_price'] / overall_mean

                    # Calculate DEMAND INDEX based on availability
                    # Lower availability = higher demand = should charge more
                    overall_avail = monthly_stats['availability'].mean()
                    # Invert availability: low availability -> high demand index
                    # Scale factor to create meaningful variation (10-20% swing)
                    monthly_stats['demand_index'] = 1 + (overall_avail - monthly_stats['availability']) * 0.5

                    # Check if price variation is too low (hosts using static pricing)
                    price_variation = (monthly_stats['price_index_raw'].max() - monthly_stats['price_index_raw'].min())
                    demand_variation = (monthly_stats['demand_index'].max() - monthly_stats['demand_index'].min())

                    # Use demand-based index if price variation < 2%
                    use_demand_index = price_variation < 0.02

                    if use_demand_index:
                        # Hosts use static pricing - use demand (availability) to estimate seasonal pricing
                        monthly_stats['price_index'] = monthly_stats['demand_index']
                        index_source = "demand (availability-based)"
                        st.warning("⚠️ **Low price variation detected** - Hosts in this market use static pricing. Using **availability/demand patterns** to estimate seasonal price adjustments.")
                    else:
                        # Normal case - use actual price variation
                        monthly_stats['price_index'] = monthly_stats['price_index_raw']
                        index_source = "actual prices"

                    monthly_stats['month_name'] = monthly_stats['month'].apply(
                        lambda x: ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December'][x]
                    )

                    # Store monthly stats in session state
                    st.session_state.monthly_price_stats = monthly_stats
                    st.session_state.seasonal_index_source = index_source

                    # Display monthly statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        # Monthly price chart - show estimated prices based on index
                        monthly_stats['estimated_price'] = overall_mean * monthly_stats['price_index']
                        fig = px.bar(monthly_stats, x='month_name', y='estimated_price',
                                    title=f"Seasonal Price Index (based on {index_source})",
                                    color='price_index',
                                    color_continuous_scale='RdYlGn_r')
                        fig.add_hline(y=overall_mean, line_dash="dash", line_color="red",
                                     annotation_text=f"Base Avg: ${overall_mean:.0f}")
                        fig.update_layout(height=350, xaxis_title="Month", yaxis_title="Estimated Price ($)")
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Price index table
                        display_stats = monthly_stats[['month_name', 'estimated_price', 'price_index', 'availability']].copy()
                        display_stats['estimated_price'] = display_stats['estimated_price'].apply(lambda x: f"${x:.2f}")
                        display_stats['price_index'] = display_stats['price_index'].apply(lambda x: f"{x:.2%}")
                        display_stats['availability'] = display_stats['availability'].apply(lambda x: f"{x:.1%}")
                        display_stats.columns = ['Month', 'Est. Price', 'Seasonal Index', 'Availability']
                        st.dataframe(display_stats, use_container_width=True, hide_index=True)

                    # Identify peak and low seasons
                    peak_month = monthly_stats.loc[monthly_stats['price_index'].idxmax(), 'month_name']
                    low_month = monthly_stats.loc[monthly_stats['price_index'].idxmin(), 'month_name']
                    peak_index = monthly_stats['price_index'].max()
                    low_index = monthly_stats['price_index'].min()

                    # Show demand analysis if using demand index
                    if use_demand_index:
                        peak_avail = monthly_stats.loc[monthly_stats['price_index'].idxmax(), 'availability']
                        low_avail = monthly_stats.loc[monthly_stats['price_index'].idxmin(), 'availability']
                        st.info(f"""
                        **🔥 High Demand (Peak):** {peak_month} ({peak_avail:.1%} availability → {(peak_index-1)*100:+.1f}% price adjustment)
                        **❄️ Low Demand:** {low_month} ({low_avail:.1%} availability → {(low_index-1)*100:+.1f}% price adjustment)
                        **📊 Seasonal Variation:** {(peak_index/low_index - 1)*100:.1f}% estimated difference between peak and low
                    """)
                    else:
                        st.info(f"""
                        **🔥 Peak Season:** {peak_month} (prices {(peak_index-1)*100:+.1f}% above average)
                        **❄️ Low Season:** {low_month} (prices {(low_index-1)*100:+.1f}% below average)
                        **📊 Price Variation:** {(peak_index/low_index - 1)*100:.1f}% difference between peak and low
                    """)
            else:
                cal_agg = None

            # REVIEWS
            progress.progress(60)
            if reviews is not None:
                with st.expander("⭐ Reviews Aggregation & Sentiment", expanded=True):
                    if 'comment_length' not in reviews.columns:
                        reviews['comment_length'] = reviews['comments'].fillna('').str.len()
                    reviews['comment_words'] = reviews['comments'].fillna('').str.split().str.len()

                    positive = ['great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'fantastic',
                                'beautiful', 'clean', 'comfortable', 'recommend', 'awesome', 'spotless', 'cozy',
                                'friendly', 'helpful', 'quiet', 'peaceful', 'convenient', 'spacious']
                    negative = ['bad', 'terrible', 'dirty', 'awful', 'worst', 'disappointing', 'problem', 'issue',
                                'noisy', 'uncomfortable', 'rude', 'broken', 'smell', 'bugs', 'cockroach']

                    st.write(f"Analyzing sentiment in {len(reviews):,} reviews...")
                    st.write(f"   Positive keywords: {len(positive)}")
                    st.write(f"   Negative keywords: {len(negative)}")

                    reviews['positive'] = reviews['comments'].fillna('').str.lower().apply(
                        lambda x: sum(1 for w in positive if w in x))
                    reviews['negative'] = reviews['comments'].fillna('').str.lower().apply(
                        lambda x: sum(1 for w in negative if w in x))
                    reviews['sentiment'] = reviews['positive'] - reviews['negative']

                    review_agg = reviews.groupby('listing_id').agg({
                        'id': 'count', 'comment_length': 'mean', 'comment_words': ['mean', 'max'],
                        'positive': 'sum', 'negative': 'sum', 'sentiment': ['sum', 'mean'], 'date': 'max'
                    }).reset_index()
                    review_agg.columns = ['listing_id', 'review_count', 'avg_comment_len', 'avg_comment_words',
                                          'max_comment_words', 'total_positive', 'total_negative',
                                          'total_sentiment', 'avg_sentiment', 'last_review']

                    review_agg['sentiment_ratio'] = review_agg['total_positive'] / (review_agg['total_negative'] + 1)
                    review_agg['last_review'] = pd.to_datetime(review_agg['last_review'], errors='coerce')
                    review_agg['days_since_review'] = (pd.Timestamp.now() - review_agg['last_review']).dt.days
                    review_agg['days_since_review'] = review_agg['days_since_review'].fillna(9999)
                    review_agg['has_recent_review'] = (review_agg['days_since_review'] < 60).astype(int)

                    st.write(f"✅ Created {len(review_agg):,} listing aggregations with 12 features")
                    st.write(f"   Avg sentiment: {review_agg['avg_sentiment'].mean():.2f}")
                    st.write(f"   Avg review count: {review_agg['review_count'].mean():.1f}")
            else:
                review_agg = None

            # MERGE
            progress.progress(85)
            with st.expander("🔗 Merging All Data", expanded=True):
                if 'price_clean' not in listings.columns:
                    listings['price_clean'] = pd.to_numeric(listings['price'].replace(r'[\$,]', '', regex=True),
                                                            errors='coerce')

                df = listings.copy()
                st.write(f"Starting with {len(df):,} listings")

                if cal_agg is not None:
                    df = df.merge(cal_agg, left_on='id', right_on='listing_id', how='left')
                    st.write(f"  + Calendar: {len(cal_agg):,} → {len(df):,} rows")

                if review_agg is not None:
                    df = df.merge(review_agg, left_on='id', right_on='listing_id', how='left', suffixes=('', '_rev'))
                    st.write(f"  + Reviews: {len(review_agg):,} → {len(df):,} rows")

                if neigh_features is not None:
                    df = df.merge(neigh_features, left_on='neighbourhood_cleansed', right_on='neighbourhood',
                                  how='left')
                    st.write(f"  + Neighbourhoods: {len(neigh_features):,} → {len(df):,} rows")

                # Fill remaining NaN with appropriate values (not 0 for everything!)
                # Review scores should use median, not 0
                review_cols = [c for c in df.columns if 'review' in c.lower() or 'rating' in c.lower()]
                for col in review_cols:
                    if col in df.columns and df[col].dtype in ['float64', 'int64'] and df[col].isnull().sum() > 0:
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        st.write(f"  • Filled {col} NaN with median: {median_val:.2f}")

                # Other numeric columns can use 0
                for col in df.columns:
                    if col not in review_cols and df[col].dtype in ['float64', 'int64'] and df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(0)

                df = df[df['price_clean'] > 0]
                st.write(f"✅ Final merged dataset: {len(df):,} rows")

            progress.progress(100)
            st.session_state.merged_data = df

            st.success(f"✅ Processing complete! {len(df):,} rows ready for feature engineering.")

# ============================================================================
# TAB 4: FEATURES
# ============================================================================
with tabs[3]:
    st.header("4️⃣ Feature Engineering")

    if st.session_state.merged_data is None:
        st.warning("⚠️ Process data first!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size %", 5, 30, 20) / 100
        with col2:
            random_state = st.number_input("Random State", 0, 100, 42)

        if st.button("⚙️ Engineer Features", type="primary", use_container_width=True):
            df = st.session_state.merged_data.copy()
            progress = st.progress(0)
            feature_log = []

            # Binary
            progress.progress(10)
            binary = {'host_is_superhost': 'is_superhost', 'host_identity_verified': 'host_verified',
                      'instant_bookable': 'instant_book', 'host_has_profile_pic': 'has_profile_pic'}
            for old, new in binary.items():
                if old in df.columns:
                    df[new] = df[old].map({'t': 1, 'f': 0}).fillna(0)
            feature_log.append(f"✅ Binary features: {len(binary)} created")

            # Categorical
            progress.progress(25)
            le_room = LabelEncoder()
            df['room_type_enc'] = le_room.fit_transform(df['room_type'].fillna('Unknown'))
            le_neigh = LabelEncoder()
            df['neighbourhood_enc'] = le_neigh.fit_transform(df['neighbourhood_cleansed'].fillna('Unknown'))
            le_prop = LabelEncoder()
            df['property_type_enc'] = le_prop.fit_transform(df['property_type'].fillna('Unknown'))

            # Save encoders to session state for prediction
            st.session_state.le_room = le_room
            st.session_state.le_neigh = le_neigh
            st.session_state.le_prop = le_prop

            feature_log.append(f"✅ Categorical encoding: 3 columns")

            # Text
            progress.progress(40)
            df['amenities_count'] = df['amenities'].fillna('[]').str.count(',') + 1
            df['desc_length'] = df['description'].fillna('').str.len()
            df['name_length'] = df['name'].fillna('').str.len()
            premium = ['pool', 'hot tub', 'gym', 'sauna', 'fireplace', 'waterfront', 'beach', 'lake', 'jacuzzi']
            df['premium_amenities'] = df['amenities'].fillna('').str.lower().apply(
                lambda x: sum(1 for a in premium if a in x))
            luxury = ['luxury', 'premium', 'waterfront', 'ocean', 'beach', 'view', 'lake', 'private', 'penthouse']
            df['has_luxury'] = df['name'].fillna('').str.lower().str.contains('|'.join(luxury)).astype(int)
            feature_log.append(f"✅ Text features: 5 created")

            # ============================================
            # LUXURY PROPERTY DETECTION (New!)
            # ============================================
            amenities_lower = df['amenities'].fillna('').str.lower()
            name_lower = df['name'].fillna('').str.lower()
            desc_lower = df['description'].fillna('').str.lower()

            # Ultra-luxury amenities (strong price indicators for $300+ properties)
            ultra_luxury = ['pool', 'hot tub', 'jacuzzi', 'sauna', 'steam room', 'wine cellar',
                          'home theater', 'theatre', 'game room', 'billiard', 'elevator', 'concierge',
                          'doorman', 'butler', 'chef', 'spa', 'massage', 'gym', 'fitness center',
                          'tennis', 'golf', 'boat', 'yacht', 'helicopter', 'private beach', 'infinity pool']
            df['ultra_luxury_count'] = amenities_lower.apply(lambda x: sum(1 for a in ultra_luxury if a in x))

            # Waterfront/view features (huge price premium)
            waterfront_terms = ['waterfront', 'beachfront', 'oceanfront', 'lakefront', 'riverfront',
                               'ocean view', 'lake view', 'water view', 'bay view', 'sea view',
                               'beach access', 'private beach', 'dock', 'marina', 'boat slip']
            df['is_waterfront'] = (amenities_lower.apply(lambda x: any(t in x for t in waterfront_terms)) |
                                   name_lower.apply(lambda x: any(t in x for t in waterfront_terms)) |
                                   desc_lower.apply(lambda x: any(t in x for t in waterfront_terms))).astype(int)

            # Property exclusivity indicators
            exclusive_terms = ['penthouse', 'mansion', 'estate', 'villa', 'castle', 'chateau',
                              'exclusive', 'prestigious', 'celebrity', 'luxury', 'high-end', 'upscale',
                              'designer', 'architect', 'custom built', 'million dollar', 'award winning']
            df['is_exclusive'] = (name_lower.apply(lambda x: any(t in x for t in exclusive_terms)) |
                                  desc_lower.apply(lambda x: any(t in x for t in exclusive_terms))).astype(int)

            # Large property indicator (high accommodates usually = higher price)
            df['is_large_property'] = (df['accommodates'] >= 8).astype(int)
            df['is_very_large'] = (df['accommodates'] >= 12).astype(int)

            # Luxury composite score (weighted sum)
            df['luxury_score'] = (df['ultra_luxury_count'] * 2 +
                                  df['is_waterfront'] * 5 +
                                  df['is_exclusive'] * 3 +
                                  df['is_large_property'] * 1 +
                                  df['is_very_large'] * 2 +
                                  df['premium_amenities'] * 1)

            # Price tier indicator (helps model recognize premium properties)
            df['luxury_tier'] = pd.cut(df['luxury_score'],
                                       bins=[-1, 0, 3, 7, 100],
                                       labels=[0, 1, 2, 3]).astype(int)

            feature_log.append(f"✅ Luxury detection: 8 features created")

            # ============================================
            # NEIGHBORHOOD PRICE CONTEXT FEATURES (New!)
            # ============================================
            st.write("Creating neighborhood price context features...")
            neigh_context_count = 0

            # Calculate neighborhood-level statistics (avoiding data leakage by using only non-target features)
            if 'neighbourhood_cleansed' in df.columns:
                # Group by neighborhood for context
                neigh_stats = df.groupby('neighbourhood_cleansed').agg({
                    'bedrooms': 'mean',
                    'accommodates': 'mean',
                    'is_superhost': 'mean',
                    'luxury_score': 'mean'
                }).reset_index()
                neigh_stats.columns = ['neighbourhood_cleansed', 'neigh_avg_beds_ctx',
                                       'neigh_avg_acc_ctx', 'neigh_superhost_rate', 'neigh_luxury_avg']

                df = df.merge(neigh_stats, on='neighbourhood_cleansed', how='left', suffixes=('', '_ctx'))

                # Price per bedroom proxy (using accommodates and bedrooms ratio as proxy)
                df['neigh_price_per_bedroom'] = df['neigh_avg_acc_ctx'] / df['neigh_avg_beds_ctx'].replace(0, 1)
                neigh_context_count += 1

                # Price per person proxy
                df['neigh_price_per_person'] = df['accommodates'] / df['neigh_avg_acc_ctx'].replace(0, 1)
                neigh_context_count += 1

                # Neighborhood luxury ratio (how luxurious is this listing vs neighborhood average)
                df['neigh_luxury_ratio'] = df['luxury_score'] / df['neigh_luxury_avg'].replace(0, 0.1)
                neigh_context_count += 1

                # Superhost premium indicator (listing is superhost in area with few superhosts = premium)
                df['neigh_superhost_premium'] = df['is_superhost'] * (1 - df['neigh_superhost_rate'])
                neigh_context_count += 1

                # Position in neighborhood (relative capacity based on accommodates)
                df['price_position_in_neigh'] = df['accommodates'] / df.groupby('neighbourhood_cleansed')['accommodates'].transform('mean').replace(0, 1)
                neigh_context_count += 1

                feature_log.append(f"✅ Neighborhood context: {neigh_context_count} features created")
            else:
                # Create dummy features if no neighborhood data
                for col in ['neigh_price_per_bedroom', 'neigh_price_per_person', 'neigh_luxury_ratio',
                           'neigh_superhost_premium', 'price_position_in_neigh']:
                    df[col] = 0
                feature_log.append("⚠️ Neighborhood context: Skipped (no neighbourhood data)")

            # Ratios
            progress.progress(55)
            df['beds_per_bedroom'] = df['beds'] / df['bedrooms'].replace(0, 1)
            df['baths_per_bedroom'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)
            df['capacity_score'] = df['accommodates'] * df['bedrooms'].replace(0, 1) * df['beds'].replace(0, 1)
            df['space_score'] = df['accommodates'] + df['bedrooms'] * 2 + df['beds'] + df['bathrooms'] * 1.5
            feature_log.append(f"✅ Ratio features: 4 created")

            # Review composite
            progress.progress(70)
            rcols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_location',
                     'review_scores_value']
            avail = [c for c in rcols if c in df.columns]
            if avail:
                df['review_composite'] = df[avail].mean(axis=1)
                df['review_min'] = df[avail].min(axis=1)
                df['review_std'] = df[avail].std(axis=1)
            feature_log.append(f"✅ Review composite: 3 created")

            # Distance
            progress.progress(85)
            center_lat, center_lon = df['latitude'].median(), df['longitude'].median()
            df['dist_center'] = np.sqrt((df['latitude'] - center_lat) ** 2 + (df['longitude'] - center_lon) ** 2)
            if 'neigh_centroid_lat' in df.columns:
                df['dist_to_neigh_center'] = np.sqrt((df['latitude'] - df['neigh_centroid_lat']) ** 2 +
                                                     (df['longitude'] - df['neigh_centroid_lon']) ** 2)
                df['neigh_density'] = df['neigh_listing_count'] / df['neigh_area'].replace(0, 1)
            feature_log.append(f"✅ Distance features: 3 created")

            # ============================================
            # ADVANCED FEATURE ENGINEERING (R² Boosters)
            # ============================================
            progress.progress(88)

            # 1. INTERACTION FEATURES - Capture non-linear relationships
            st.write("Creating interaction features...")
            interaction_count = 0

            # Bedrooms × Location quality (high bedrooms in good location = premium)
            if 'bedrooms' in df.columns and 'review_scores_location' in df.columns:
                df['bedrooms_x_location'] = df['bedrooms'] * df['review_scores_location']
                interaction_count += 1

            # Accommodates × Superhost (large capacity + superhost = premium)
            if 'accommodates' in df.columns and 'is_superhost' in df.columns:
                df['accommodates_x_superhost'] = df['accommodates'] * df['is_superhost']
                interaction_count += 1

            # Bedrooms × Bathrooms (luxury indicator)
            if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
                df['bedrooms_x_bathrooms'] = df['bedrooms'] * df['bathrooms']
                interaction_count += 1

            # Reviews × Rating (social proof strength)
            if 'number_of_reviews' in df.columns and 'review_scores_rating' in df.columns:
                df['reviews_x_rating'] = np.log1p(df['number_of_reviews']) * df['review_scores_rating']
                interaction_count += 1

            # Accommodates × Instant book (convenience for groups)
            if 'accommodates' in df.columns and 'instant_book' in df.columns:
                df['accommodates_x_instant'] = df['accommodates'] * df['instant_book']
                interaction_count += 1

            # Room type × Bedrooms (entire home with many bedrooms = premium)
            if 'room_type_enc' in df.columns and 'bedrooms' in df.columns:
                df['roomtype_x_bedrooms'] = df['room_type_enc'] * df['bedrooms']
                interaction_count += 1

            # Amenities × Rating (well-equipped + good reviews = premium)
            if 'amenities_count' in df.columns and 'review_scores_rating' in df.columns:
                df['amenities_x_rating'] = df['amenities_count'] * df['review_scores_rating']
                interaction_count += 1

            feature_log.append(f"✅ Interaction features: {interaction_count} created")

            # 2. LOCATION CLUSTERING - Create micro-neighborhoods
            st.write("Creating location clusters...")
            if 'latitude' in df.columns and 'longitude' in df.columns:
                # Determine optimal number of clusters (sqrt of listings, min 5, max 50)
                n_clusters = min(max(int(np.sqrt(len(df))), 5), 50)

                coords = df[['latitude', 'longitude']].dropna()
                if len(coords) > n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df['location_cluster'] = -1  # Default for missing coords
                    df.loc[coords.index, 'location_cluster'] = kmeans.fit_predict(coords)

                    # Calculate cluster statistics (average price per cluster would be leakage, so use other metrics)
                    cluster_stats = df.groupby('location_cluster').agg({
                        'review_scores_rating': 'mean',
                        'accommodates': 'mean',
                        'number_of_reviews': 'mean',
                        'is_superhost': 'mean'
                    }).reset_index()
                    cluster_stats.columns = ['location_cluster', 'cluster_avg_rating', 'cluster_avg_accommodates',
                                             'cluster_avg_reviews', 'cluster_superhost_ratio']

                    # Merge cluster stats back
                    df = df.merge(cluster_stats, on='location_cluster', how='left')

                    # Distance to cluster center
                    cluster_centers = kmeans.cluster_centers_
                    df['dist_to_cluster_center'] = 0.0
                    for i in range(n_clusters):
                        mask = df['location_cluster'] == i
                        if mask.sum() > 0:
                            center_lat, center_lon = cluster_centers[i]
                            df.loc[mask, 'dist_to_cluster_center'] = np.sqrt(
                                (df.loc[mask, 'latitude'] - center_lat) ** 2 +
                                (df.loc[mask, 'longitude'] - center_lon) ** 2
                            )

                    feature_log.append(f"✅ Location clustering: {n_clusters} clusters, 5 features created")
                else:
                    feature_log.append("⚠️ Location clustering: Not enough data points")

            # 3. ENHANCED TEXT FEATURES
            st.write("Creating enhanced text features...")
            text_feature_count = 0

            # Description quality indicators
            if 'description' in df.columns:
                desc = df['description'].fillna('')
                # Word count
                df['desc_word_count'] = desc.str.split().str.len()
                # Sentence count (approximate)
                df['desc_sentence_count'] = desc.str.count(r'[.!?]+')
                # Average word length (quality indicator)
                df['desc_avg_word_len'] = desc.apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
                # Has detailed description
                df['has_detailed_desc'] = (df['desc_word_count'] > 100).astype(int)
                text_feature_count += 4

            # Amenities parsing (specific high-value amenities)
            if 'amenities' in df.columns:
                amenities_lower = df['amenities'].fillna('').str.lower()

                # High-value amenity categories
                df['has_parking'] = amenities_lower.str.contains('parking|garage').astype(int)
                df['has_wifi'] = amenities_lower.str.contains('wifi|internet|wi-fi').astype(int)
                df['has_kitchen'] = amenities_lower.str.contains('kitchen|stove|oven|microwave').astype(int)
                df['has_ac'] = amenities_lower.str.contains('air condition|ac|a/c|conditioning').astype(int)
                df['has_washer'] = amenities_lower.str.contains('washer|laundry|dryer').astype(int)
                df['has_tv'] = amenities_lower.str.contains('tv|television|netflix|cable').astype(int)
                df['has_workspace'] = amenities_lower.str.contains('workspace|desk|laptop|office').astype(int)
                df['has_pets_allowed'] = amenities_lower.str.contains('pet|dog|cat').astype(int)

                # Premium amenity score (sum of high-value amenities)
                df['premium_amenity_score'] = (df['has_parking'] + df['has_wifi'] + df['has_kitchen'] +
                                                df['has_ac'] + df['has_washer'] + df['has_tv'] +
                                                df['has_workspace'] + df['premium_amenities'])
                text_feature_count += 9

            # Name analysis
            if 'name' in df.columns:
                name_lower = df['name'].fillna('').str.lower()
                # Location mentions in name
                df['name_has_location'] = name_lower.str.contains('downtown|central|beach|lake|mountain|view|near').astype(int)
                # Size mentions
                df['name_has_size'] = name_lower.str.contains('spacious|large|cozy|tiny|studio|loft').astype(int)
                # Special features in name
                df['name_has_special'] = name_lower.str.contains('modern|new|renovated|historic|unique|charming').astype(int)
                text_feature_count += 3

            feature_log.append(f"✅ Enhanced text features: {text_feature_count} created")

            # 4. POLYNOMIAL FEATURES (for top predictors only)
            st.write("Creating polynomial features...")
            poly_cols = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
            poly_cols = [c for c in poly_cols if c in df.columns]

            if len(poly_cols) >= 2:
                # Square terms (diminishing returns for size)
                for col in poly_cols:
                    df[f'{col}_squared'] = df[col] ** 2

                # Log transforms (for skewed distributions)
                for col in poly_cols:
                    df[f'{col}_log'] = np.log1p(df[col])

                feature_log.append(f"✅ Polynomial features: {len(poly_cols) * 2} created (squares + logs)")

            # 5. RATIO FEATURES (additional)
            st.write("Creating additional ratio features...")
            ratio_count = 0

            # Guest per bed ratio
            if 'accommodates' in df.columns and 'beds' in df.columns:
                df['guests_per_bed'] = df['accommodates'] / df['beds'].replace(0, 1)
                ratio_count += 1

            # Review density (reviews per month normalized by listing age proxy)
            if 'reviews_per_month' in df.columns and 'number_of_reviews' in df.columns:
                df['review_velocity'] = df['reviews_per_month'] * np.log1p(df['number_of_reviews'])
                ratio_count += 1

            # Availability ratio (scarcity indicator)
            if 'cal_avail_rate' in df.columns:
                df['scarcity_score'] = 1 - df['cal_avail_rate']  # Lower availability = higher scarcity
                ratio_count += 1

            # Quality score (composite of all review dimensions)
            review_cols_all = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_location',
                              'review_scores_value', 'review_scores_checkin', 'review_scores_communication']
            available_review_cols = [c for c in review_cols_all if c in df.columns]
            if len(available_review_cols) > 0:
                df['quality_score'] = df[available_review_cols].mean(axis=1)
                ratio_count += 1

            feature_log.append(f"✅ Additional ratio features: {ratio_count} created")

            # Normalize review scores (handle both 0-5 and 0-100 scales)
            review_score_cols = ['review_scores_rating', 'review_scores_cleanliness',
                                 'review_scores_location', 'review_scores_value',
                                 'review_scores_checkin', 'review_scores_communication', 'review_scores_accuracy']
            for col in review_score_cols:
                if col in df.columns:
                    # If max > 10, it's on 0-100 scale, convert to 0-5
                    col_max = df[col].max()
                    if col_max > 10:
                        df[col] = df[col] / 20  # Convert 0-100 to 0-5
                        feature_log.append(f"✅ Normalized {col} from 0-100 to 0-5 scale")

            # Recalculate review composite after normalization
            rcols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_value']
            avail = [c for c in rcols if c in df.columns]
            if avail:
                df['review_composite'] = df[avail].mean(axis=1)
                df['review_min'] = df[avail].min(axis=1)
                df['review_std'] = df[avail].std(axis=1)

            # Final feature list
            # NOTE: REMOVED calendar price features (cal_price_mean, cal_price_std, etc.)
            # to prevent data leakage - these are derived from prices and would predict price using price!
            features = [
                # Core property features (most predictive)
                'accommodates', 'bedrooms', 'beds', 'bathrooms',
                'latitude', 'longitude', 'neighbourhood_enc', 'dist_center',
                'room_type_enc', 'property_type_enc',
                # Host features
                'is_superhost', 'host_verified', 'has_profile_pic', 'host_listings_count',
                'host_response_rate', 'host_acceptance_rate',
                # Listing features
                'instant_book', 'minimum_nights', 'amenities_count', 'desc_length', 'name_length',
                'has_luxury', 'premium_amenities',
                # Review scores (normalized to 0-5)
                'review_scores_rating', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_value',
                'review_composite', 'review_min', 'review_std', 'number_of_reviews', 'reviews_per_month',
                # Derived ratios
                'beds_per_bedroom', 'baths_per_bedroom', 'capacity_score', 'space_score',
                # Calendar availability only (NOT price features - those cause data leakage!)
                'cal_avail_rate', 'high_demand',
                # Review sentiment
                'review_count', 'avg_comment_len', 'avg_comment_words', 'max_comment_words',
                'total_positive', 'total_negative', 'avg_sentiment', 'sentiment_ratio',
                'days_since_review', 'has_recent_review',
                # Neighbourhood features
                'neigh_centroid_lon', 'neigh_centroid_lat', 'neigh_area', 'neigh_perimeter',
                'neigh_group_enc', 'neigh_listing_count', 'neigh_density',
                'neigh_avg_rating', 'neigh_avg_accommodates', 'neigh_avg_bedrooms', 'neigh_avg_reviews',
                'dist_to_neigh_center',
                # === NEW ADVANCED FEATURES ===
                # Interaction features (capture non-linear relationships)
                'bedrooms_x_location', 'accommodates_x_superhost', 'bedrooms_x_bathrooms',
                'reviews_x_rating', 'accommodates_x_instant', 'roomtype_x_bedrooms', 'amenities_x_rating',
                # Location clustering features
                'location_cluster', 'cluster_avg_rating', 'cluster_avg_accommodates',
                'cluster_avg_reviews', 'cluster_superhost_ratio', 'dist_to_cluster_center',
                # Enhanced text features
                'desc_word_count', 'desc_sentence_count', 'desc_avg_word_len', 'has_detailed_desc',
                'has_parking', 'has_wifi', 'has_kitchen', 'has_ac', 'has_washer',
                'has_tv', 'has_workspace', 'has_pets_allowed', 'premium_amenity_score',
                'name_has_location', 'name_has_size', 'name_has_special',
                # Polynomial features
                'accommodates_squared', 'bedrooms_squared', 'beds_squared', 'bathrooms_squared',
                'accommodates_log', 'bedrooms_log', 'beds_log', 'bathrooms_log',
                # Additional ratio features
                'guests_per_bed', 'review_velocity', 'scarcity_score', 'quality_score',
                # === LUXURY PROPERTY DETECTION FEATURES (NEW!) ===
                'ultra_luxury_count', 'is_waterfront', 'is_exclusive',
                'is_large_property', 'is_very_large', 'luxury_score', 'luxury_tier',
                # === NEIGHBORHOOD PRICE CONTEXT FEATURES (NEW!) ===
                'neigh_price_per_bedroom', 'neigh_price_per_person', 'neigh_luxury_ratio',
                'neigh_superhost_premium', 'price_position_in_neigh'
            ]
            features = [f for f in features if f in df.columns]
            st.session_state.features = features

            # Prepare
            df_model = df[features + ['price_clean']].copy()
            for col in features:
                df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)
            df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

            X = df_model[features]
            y = df_model['price_clean']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.X_train_scaled = X_train_scaled
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.scaler = scaler

            progress.progress(100)

            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Features", len(features))
            with col2:
                st.metric("📊 Train", f"{len(X_train):,}")
            with col3:
                st.metric("🧪 Test", f"{len(X_test):,}")

            with st.expander("📋 Feature Engineering Log"):
                for log in feature_log:
                    st.write(log)

            with st.expander("📊 All Features by Category"):
                categories = {
                    'Basic (4)': ['accommodates', 'bedrooms', 'beds', 'bathrooms'],
                    'Location (4)': ['latitude', 'longitude', 'neighbourhood_enc', 'dist_center'],
                    'Host (6)': ['is_superhost', 'host_verified', 'has_profile_pic', 'host_listings_count',
                                 'host_response_rate', 'host_acceptance_rate'],
                    'Listing (7)': ['room_type_enc', 'property_type_enc', 'instant_book', 'minimum_nights',
                                    'amenities_count', 'desc_length', 'name_length', 'has_luxury', 'premium_amenities'],
                    'Calendar (13)': [f for f in features if
                                      'cal_' in f or f in ['price_range', 'price_iqr', 'price_volatility',
                                                           'dynamic_pricing', 'high_demand']],
                    'Reviews (18)': [f for f in features if
                                     any(x in f for x in ['review', 'sentiment', 'comment', 'positive', 'negative'])],
                    'Neighbourhood (12)': [f for f in features if 'neigh' in f or f == 'dist_to_neigh_center'],
                }
                for cat, feats in categories.items():
                    present = [f for f in feats if f in features]
                    st.write(f"**{cat}**: {', '.join(present)}")

            st.success("✅ Features ready!")

# ============================================================================
# TAB 5-8: MODEL SELECTION, TRAINING, RESULTS, PREDICT
# (Same as before - keeping them shorter for space)
# ============================================================================
with tabs[4]:
    st.header("5️⃣ Model Selection")

    # Build categories
    categories = {}
    all_model_names = []
    for name, info in ALL_MODELS.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)
        all_model_names.append(name)

    # Best and fast model lists
    best_models = ['LightGBM', 'XGBoost', 'CatBoost', 'Random Forest']
    fast_models = ['Linear Regression', 'Decision Tree', 'Random Forest']

    selected_models = []

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✅ Select All", use_container_width=True, key="btn_select_all"):
            for name in all_model_names:
                st.session_state[f"m_{name}"] = True
            st.rerun()
    with col2:
        if st.button("❌ Clear All", use_container_width=True, key="btn_clear_all"):
            for name in all_model_names:
                st.session_state[f"m_{name}"] = False
            st.rerun()
    with col3:
        if st.button("🚀 Best Only", use_container_width=True, key="btn_best"):
            for name in all_model_names:
                st.session_state[f"m_{name}"] = name in best_models
            st.rerun()
    with col4:
        if st.button("⚡ Fast Only", use_container_width=True, key="btn_fast"):
            for name in all_model_names:
                st.session_state[f"m_{name}"] = name in fast_models
            st.rerun()

    for cat, models in categories.items():
        st.markdown(f"#### {cat}")
        cols = st.columns(3)
        for i, model_name in enumerate(models):
            with cols[i % 3]:
                if st.checkbox(model_name, key=f"m_{model_name}"):
                    selected_models.append(model_name)

    st.session_state.selected_models = selected_models
    st.info(f"Selected: {len(selected_models)} models")

with tabs[5]:
    st.header("6️⃣ Training")

    if st.session_state.X_train is None:
        st.warning("⚠️ Complete feature engineering first!")
    elif not st.session_state.get('selected_models'):
        st.warning("⚠️ Select models first!")
    else:
        # Training Options
        st.markdown("### ⚙️ Training Options")

        # Basic Options
        st.markdown("#### 📋 Basic Options")
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        with opt_col1:
            use_tuned = st.checkbox("🎯 Use Tuned Parameters", value=False,
                                    help="Use optimized hyperparameters for better accuracy (recommended)")
        with opt_col2:
            log_transform = st.checkbox("📈 Log Transform Target", value=False,
                                        help="Apply log transformation to price - helps with skewed data")
        with opt_col3:
            create_ensemble = st.checkbox("🔗 Create Ensemble", value=False,
                                          help="Average predictions from top 3 models for more stable results")

        # Advanced Options (for 88-92% accuracy)
        st.markdown("#### 🚀 Advanced Options (Target: 88-92% Accuracy)")
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            target_clipping = st.checkbox("✂️ Target Clipping", value=False,
                                          help="Cap extreme prices at percentile bounds - removes hard-to-predict outliers")
            feature_selection = st.checkbox("🔧 Feature Selection", value=False,
                                            help="Keep only features with >= 0.5% importance (keeps ~40-50 features)")
        with adv_col2:
            stacking_ensemble = st.checkbox("📚 Stacking Ensemble", value=False,
                                            help="Use model predictions as features for a meta-learner (most powerful)")
            aggressive_tuning = st.checkbox("⚡ Aggressive Tuning", value=False,
                                            help="Even more iterations and optimized parameters (slower but better)")

        # Price-Segmented Models (New option for handling extreme prices better)
        st.markdown("#### 💰 Price Segment Options")
        seg_col1, seg_col2 = st.columns(2)
        with seg_col1:
            price_segmented = st.checkbox("🎯 Price-Segmented Models", value=False,
                                          help="Train separate models for Budget ($0-100), Standard ($100-300), Premium ($300+)")
        with seg_col2:
            hard_price_cap = st.checkbox("🔒 Hard Price Cap", value=False,
                                         help="Remove listings above a fixed price threshold (recommended for better accuracy)")

        # Clipping/Cap settings
        clip_percentile = 3  # More aggressive default
        min_price_cap = 10   # Default min cap
        max_price_cap = 500  # Default max cap
        if target_clipping:
            clip_percentile = st.slider("Clipping Percentile", min_value=1, max_value=10, value=3,
                                        help="Remove prices below X% and above (100-X)%. Lower = more aggressive.")
        if hard_price_cap:
            cap_col1, cap_col2 = st.columns(2)
            with cap_col1:
                min_price_cap = st.slider("Minimum Price Cap ($)", min_value=0, max_value=100, value=10, step=5,
                                          help="Remove all listings priced below this amount.")
            with cap_col2:
                max_price_cap = st.slider("Maximum Price Cap ($)", min_value=200, max_value=1000, value=500, step=50,
                                          help="Remove all listings priced above this amount.")

        # Show warnings for conflicting options
        if log_transform and price_segmented:
            st.warning("⚠️ **Conflict:** Log Transform compresses price ranges, reducing the effectiveness of Price-Segmented Models. Consider disabling Log Transform.")

        # Show info about selected options
        all_options = [use_tuned, log_transform, create_ensemble, target_clipping, feature_selection, stacking_ensemble, aggressive_tuning]
        if any(all_options):
            options_info = []
            if use_tuned:
                options_info.append("**Tuned Params**")
            if log_transform:
                options_info.append("**Log Transform**")
            if create_ensemble:
                options_info.append("**Ensemble**")
            if target_clipping:
                options_info.append(f"**Clipping ({clip_percentile}%-{100-clip_percentile}%)**")
            if feature_selection:
                options_info.append("**Feature Selection**")
            if stacking_ensemble:
                options_info.append("**Stacking**")
            if aggressive_tuning:
                options_info.append("**Aggressive Tuning**")
            st.info("📋 Active: " + " | ".join(options_info))

            # Expected improvement estimate
            expected_boost = 0
            if use_tuned: expected_boost += 2
            if log_transform: expected_boost += 2
            if create_ensemble: expected_boost += 1
            if target_clipping: expected_boost += 2
            if feature_selection: expected_boost += 1
            if stacking_ensemble: expected_boost += 3
            if aggressive_tuning: expected_boost += 2
            st.success(f"🎯 Expected accuracy boost: **+{min(expected_boost, 12)}%** (results may vary)")

        st.markdown("---")

        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            results = {}
            trained = {}
            progress = st.progress(0)

            # Select model configuration based on user choice
            if aggressive_tuning:
                active_models = AGGRESSIVE_MODELS
                st.info("⚡ Using **aggressive hyperparameters** for maximum accuracy (slower)")
            elif use_tuned:
                active_models = TUNED_MODELS
                st.info("🎯 Using **tuned hyperparameters** for better accuracy")
            else:
                active_models = DEFAULT_MODELS

            # Prepare target variable
            y_train = st.session_state.y_train.copy()
            y_test = st.session_state.y_test.copy()
            X_train = st.session_state.X_train.copy()
            X_test = st.session_state.X_test.copy()
            X_train_scaled = st.session_state.X_train_scaled.copy()
            X_test_scaled = st.session_state.X_test_scaled.copy()
            features = st.session_state.features.copy()

            # =====================================================
            # ADVANCED OPTION 1: TARGET CLIPPING
            # =====================================================
            if target_clipping:
                st.write(f"✂️ Applying target clipping at {clip_percentile}%-{100-clip_percentile}% percentile...")
                lower_bound = y_train.quantile(clip_percentile / 100)
                upper_bound = y_train.quantile((100 - clip_percentile) / 100)

                # Count how many will be clipped
                below_count = (y_train < lower_bound).sum()
                above_count = (y_train > upper_bound).sum()

                # Clip the training target (not test - we want to evaluate on real data)
                y_train = y_train.clip(lower=lower_bound, upper=upper_bound)

                st.success(f"✅ Clipped {below_count + above_count} extreme values (${lower_bound:.2f} - ${upper_bound:.2f})")

            # =====================================================
            # ADVANCED OPTION: HARD PRICE CAP (Remove extreme outliers)
            # =====================================================
            if hard_price_cap:
                st.write(f"🔒 Applying hard price cap: ${min_price_cap} - ${max_price_cap}...")

                # Remove listings outside the price range from BOTH train and test
                train_mask = (y_train >= min_price_cap) & (y_train <= max_price_cap)
                test_mask = (y_test >= min_price_cap) & (y_test <= max_price_cap)

                train_removed = (~train_mask).sum()
                test_removed = (~test_mask).sum()

                # Count low vs high removals for info
                train_low = (y_train < min_price_cap).sum()
                train_high = (y_train > max_price_cap).sum()
                test_low = (y_test < min_price_cap).sum()
                test_high = (y_test > max_price_cap).sum()

                # Apply filters
                y_train = y_train[train_mask]
                X_train = X_train[train_mask]
                X_train_scaled = X_train_scaled[train_mask]

                y_test = y_test[test_mask]
                X_test = X_test[test_mask]
                X_test_scaled = X_test_scaled[test_mask]

                st.success(f"✅ Removed {train_removed} train ({train_low} below ${min_price_cap}, {train_high} above ${max_price_cap}) and {test_removed} test listings")

                # CRITICAL: Update session state with filtered data
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled

            if log_transform:
                st.info("📈 Applying **log transform** to target variable")
                y_train = np.log1p(y_train)  # log(1 + y) to handle zeros

            # =====================================================
            # ADVANCED OPTION 2: FEATURE SELECTION
            # =====================================================
            selected_features = features.copy()
            if feature_selection:
                st.write("🔧 Performing feature selection...")

                # Use Random Forest to get feature importances
                from sklearn.ensemble import RandomForestRegressor
                fs_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                fs_model.fit(X_train, y_train)

                # Get feature importances
                importances = pd.DataFrame({
                    'feature': features,
                    'importance': fs_model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Keep features with importance >= 0.5% (less aggressive to retain more useful features)
                importance_threshold = 0.005
                important_features = importances[importances['importance'] >= importance_threshold]['feature'].tolist()

                # Ensure we keep at least 40 features (was 20 - too aggressive)
                if len(important_features) < 40:
                    important_features = importances.head(40)['feature'].tolist()

                removed_features = len(features) - len(important_features)
                selected_features = important_features

                # Update training data with selected features only
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
                X_train_scaled = X_train_scaled[selected_features]
                X_test_scaled = X_test_scaled[selected_features]

                st.success(f"✅ Feature selection: kept {len(selected_features)}/{len(features)} features (removed {removed_features} low-importance)")

                # Show top 10 features
                with st.expander("📊 Top Features by Importance"):
                    top_10 = importances.head(10)
                    for _, row in top_10.iterrows():
                        st.write(f"  • **{row['feature']}**: {row['importance']:.3f}")

            # Store training settings for later use in predictions
            st.session_state.training_settings = {
                'use_tuned': use_tuned,
                'log_transform': log_transform,
                'create_ensemble': create_ensemble,
                'selected_features': selected_features
            }

            # CRITICAL: Update session state with feature-selected data
            # This ensures Predict tab uses the same features as training
            if feature_selection:
                st.session_state.features = selected_features
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
                # Re-fit scaler on selected features only
                scaler = StandardScaler()
                st.session_state.X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train), columns=selected_features, index=X_train.index
                )
                st.session_state.X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test), columns=selected_features, index=X_test.index
                )
                st.session_state.scaler = scaler
                X_train_scaled = st.session_state.X_train_scaled
                X_test_scaled = st.session_state.X_test_scaled

            for i, name in enumerate(st.session_state.selected_models):
                st.write(f"Training {name}{'  (tuned)' if use_tuned else ''}...")

                # Get model config from active models
                if name not in active_models:
                    st.warning(f"⚠️ {name} not available in selected configuration, skipping...")
                    continue

                info = active_models[name]
                model = type(info['model'])(**info['model'].get_params())

                # Use local variables (which may have been modified by feature selection)
                X_tr = X_train_scaled if info['scale'] else X_train
                X_te = X_test_scaled if info['scale'] else X_test

                try:
                    model.fit(X_tr, y_train)
                    pred = model.predict(X_te)

                    # Reverse log transform if applied
                    if log_transform:
                        pred = np.expm1(pred)  # exp(y) - 1 to reverse log1p
                        pred = np.maximum(pred, 0)  # Ensure non-negative

                    r2 = r2_score(y_test, pred)
                    mae = mean_absolute_error(y_test, pred)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))

                    # Calculate MAPE (Mean Absolute Percentage Error)
                    actual = y_test
                    # Avoid division by zero - exclude zero actual values
                    mask = actual != 0
                    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
                    prediction_accuracy = 100 - mape  # Your target: 80-100%

                    results[name] = {
                        'R2': r2,
                        'Accuracy': r2 * 100,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'Prediction_Accuracy': prediction_accuracy,
                        'predictions': pred
                    }
                    trained[name] = {
                        'model': model,
                        'needs_scale': info['scale'],
                        'r2': r2,
                        'mape': mape,
                        'prediction_accuracy': prediction_accuracy,
                        'mae': mae,
                        'log_transform': log_transform
                    }
                except Exception as e:
                    results[name] = {'R2': 0, 'error': str(e)}
                    st.warning(f"⚠️ {name} failed: {str(e)[:50]}...")

                progress.progress((i + 1) / len(st.session_state.selected_models))

            # Create ensemble if selected and we have at least 2 successful models
            if create_ensemble and len([r for r in results.values() if 'error' not in r]) >= 2:
                st.write("🔗 Creating ensemble from top models...")

                # Get top 3 models by R2
                successful_results = [(k, v) for k, v in results.items() if 'error' not in v]
                sorted_by_r2 = sorted(successful_results, key=lambda x: x[1]['R2'], reverse=True)
                top_models = sorted_by_r2[:min(3, len(sorted_by_r2))]

                # Calculate weighted average predictions
                weights = [m[1]['R2'] for m in top_models]
                weight_sum = sum(weights)
                weights = [w / weight_sum for w in weights]  # Normalize

                ensemble_pred = np.zeros(len(y_test))
                for (model_name, _), weight in zip(top_models, weights):
                    ensemble_pred += results[model_name]['predictions'] * weight

                # Calculate ensemble metrics
                r2_ens = r2_score(y_test, ensemble_pred)
                mae_ens = mean_absolute_error(y_test, ensemble_pred)
                rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                mask_ens = y_test != 0
                mape_ens = np.mean(np.abs((y_test[mask_ens] - ensemble_pred[mask_ens]) / y_test[mask_ens])) * 100
                pred_acc_ens = 100 - mape_ens

                ensemble_name = f"🔗 Ensemble ({', '.join([m[0] for m in top_models])})"
                results[ensemble_name] = {
                    'R2': r2_ens,
                    'Accuracy': r2_ens * 100,
                    'MAE': mae_ens,
                    'RMSE': rmse_ens,
                    'MAPE': mape_ens,
                    'Prediction_Accuracy': pred_acc_ens,
                    'predictions': ensemble_pred
                }
                trained[ensemble_name] = {
                    'model': None,  # Ensemble doesn't have a single model
                    'ensemble_models': [m[0] for m in top_models],
                    'ensemble_weights': weights,
                    'needs_scale': False,
                    'r2': r2_ens,
                    'mape': mape_ens,
                    'prediction_accuracy': pred_acc_ens,
                    'mae': mae_ens,
                    'log_transform': log_transform
                }

                st.success(f"✅ Ensemble created with R²: {r2_ens:.2%} | Prediction Accuracy: {pred_acc_ens:.1f}%")

            # =====================================================
            # ADVANCED OPTION 4: STACKING ENSEMBLE
            # =====================================================
            if stacking_ensemble and len([k for k, v in results.items() if 'error' not in v and '🔗' not in k]) >= 2:
                st.write("📚 Building Stacking Ensemble (meta-learner)...")

                # Get base model predictions (excluding the regular ensemble)
                base_models = [(k, v) for k, v in results.items() if 'error' not in v and '🔗' not in k]

                # Create meta-features from base model predictions
                meta_train_features = np.column_stack([
                    trained[model_name]['model'].predict(
                        X_train_scaled if trained[model_name].get('needs_scale', False) else X_train
                    ) for model_name, _ in base_models
                ])

                meta_test_features = np.column_stack([
                    results[model_name]['predictions'] for model_name, _ in base_models
                ])

                # Train Ridge regression as meta-learner (robust and fast)
                from sklearn.linear_model import Ridge
                meta_learner = Ridge(alpha=1.0)
                meta_learner.fit(meta_train_features, y_train)

                # Get stacking predictions
                stacking_pred = meta_learner.predict(meta_test_features)

                # Calculate stacking metrics
                r2_stack = r2_score(y_test, stacking_pred)
                mae_stack = mean_absolute_error(y_test, stacking_pred)
                rmse_stack = np.sqrt(mean_squared_error(y_test, stacking_pred))
                mask_stack = y_test != 0
                mape_stack = np.mean(np.abs((y_test[mask_stack] - stacking_pred[mask_stack]) / y_test[mask_stack])) * 100
                pred_acc_stack = 100 - mape_stack

                stacking_name = f"📚 Stacking ({len(base_models)} models)"
                results[stacking_name] = {
                    'R2': r2_stack,
                    'Accuracy': r2_stack * 100,
                    'MAE': mae_stack,
                    'RMSE': rmse_stack,
                    'MAPE': mape_stack,
                    'Prediction_Accuracy': pred_acc_stack,
                    'predictions': stacking_pred
                }
                trained[stacking_name] = {
                    'model': meta_learner,
                    'base_models': [m[0] for m in base_models],
                    'needs_scale': False,
                    'is_stacking': True,
                    'r2': r2_stack,
                    'mape': mape_stack,
                    'prediction_accuracy': pred_acc_stack,
                    'mae': mae_stack,
                    'log_transform': log_transform
                }

                st.success(f"✅ Stacking Ensemble: R²: {r2_stack:.2%} | Prediction Accuracy: {pred_acc_stack:.1f}%")

            # =====================================================
            # ADVANCED OPTION 5: PRICE-SEGMENTED MODELS
            # =====================================================
            if price_segmented:
                st.write("🎯 Training Price-Segmented Models...")

                # Define price segments
                segment_bounds = {
                    'Budget': (0, 100),
                    'Standard': (100, 300),
                    'Premium': (300, float('inf'))
                }

                # Get the best performing base model for segmented training
                successful_base_models = [(k, v) for k, v in results.items()
                                         if 'error' not in v and '🔗' not in k and '📚' not in k]
                if successful_base_models:
                    best_base_model_name = sorted(successful_base_models,
                                                  key=lambda x: x[1]['Prediction_Accuracy'],
                                                  reverse=True)[0][0]
                    best_base_config = active_models.get(best_base_model_name)

                    if best_base_config:
                        segment_models = {}
                        segment_stats = {}

                        # Train a model for each segment
                        for segment_name, (low, high) in segment_bounds.items():
                            # Filter training data by price segment
                            if log_transform:
                                # If log transform was applied, we need to use original y_train
                                y_train_orig = st.session_state.y_train.copy()
                                if target_clipping:
                                    y_train_orig = y_train_orig.clip(lower=lower_bound, upper=upper_bound)
                                segment_mask_train = (y_train_orig >= low) & (y_train_orig < high)
                            else:
                                segment_mask_train = (y_train >= low) & (y_train < high)

                            segment_mask_test = (y_test >= low) & (y_test < high)

                            n_train_segment = segment_mask_train.sum()
                            n_test_segment = segment_mask_test.sum()

                            if n_train_segment < 50:
                                st.warning(f"⚠️ {segment_name} segment has only {n_train_segment} samples, skipping...")
                                continue

                            st.write(f"  Training {segment_name} model ({n_train_segment:,} train, {n_test_segment:,} test)...")

                            # Get segment data
                            X_train_seg = X_train[segment_mask_train]
                            y_train_seg = y_train[segment_mask_train]
                            X_train_scaled_seg = X_train_scaled[segment_mask_train]

                            # Create and train model
                            seg_model = type(best_base_config['model'])(**best_base_config['model'].get_params())
                            X_tr_seg = X_train_scaled_seg if best_base_config['scale'] else X_train_seg

                            try:
                                seg_model.fit(X_tr_seg, y_train_seg)

                                # Predict on segment test data
                                X_test_seg = X_test[segment_mask_test]
                                X_test_scaled_seg = X_test_scaled[segment_mask_test]
                                y_test_seg = y_test[segment_mask_test]

                                X_te_seg = X_test_scaled_seg if best_base_config['scale'] else X_test_seg
                                pred_seg = seg_model.predict(X_te_seg)

                                # Reverse log transform if applied
                                if log_transform:
                                    pred_seg = np.expm1(pred_seg)
                                    pred_seg = np.maximum(pred_seg, 0)

                                # Calculate segment metrics
                                if n_test_segment > 0:
                                    r2_seg = r2_score(y_test_seg, pred_seg)
                                    mae_seg = mean_absolute_error(y_test_seg, pred_seg)
                                    mask_seg = y_test_seg != 0
                                    mape_seg = np.mean(np.abs((y_test_seg[mask_seg] - pred_seg[mask_seg]) / y_test_seg[mask_seg])) * 100
                                    pred_acc_seg = 100 - mape_seg

                                    segment_models[segment_name] = {
                                        'model': seg_model,
                                        'bounds': (low, high),
                                        'needs_scale': best_base_config['scale'],
                                        'n_samples': n_train_segment
                                    }

                                    segment_stats[segment_name] = {
                                        'R2': r2_seg,
                                        'MAPE': mape_seg,
                                        'MAE': mae_seg,
                                        'Prediction_Accuracy': pred_acc_seg,
                                        'n_train': n_train_segment,
                                        'n_test': n_test_segment
                                    }

                                    st.write(f"    ✅ {segment_name}: R²={r2_seg:.2%}, Acc={pred_acc_seg:.1f}%, MAE=${mae_seg:.2f}")
                            except Exception as e:
                                st.warning(f"    ⚠️ {segment_name} failed: {str(e)[:50]}")

                        # Create combined segmented prediction
                        if len(segment_models) >= 2:
                            st.write("  Creating combined segmented predictions...")

                            # Predict using appropriate segment model based on predicted price range
                            # First, get initial predictions from best base model to determine segments
                            base_model_info = trained[best_base_model_name]
                            X_te_base = X_test_scaled if base_model_info['needs_scale'] else X_test
                            initial_pred = base_model_info['model'].predict(X_te_base)

                            if log_transform:
                                initial_pred = np.expm1(initial_pred)
                                initial_pred = np.maximum(initial_pred, 0)

                            # Now predict using segment-specific models
                            segmented_pred = np.zeros(len(y_test))
                            segment_assignment = np.zeros(len(y_test), dtype=int)

                            for i, pred_val in enumerate(initial_pred):
                                # Determine segment based on initial prediction
                                if pred_val < 100 and 'Budget' in segment_models:
                                    segment = 'Budget'
                                    segment_assignment[i] = 0
                                elif pred_val < 300 and 'Standard' in segment_models:
                                    segment = 'Standard'
                                    segment_assignment[i] = 1
                                elif 'Premium' in segment_models:
                                    segment = 'Premium'
                                    segment_assignment[i] = 2
                                else:
                                    # Fallback to base model prediction
                                    segmented_pred[i] = initial_pred[i]
                                    continue

                                # Get segment model prediction
                                seg_info = segment_models[segment]
                                X_single = X_test_scaled.iloc[[i]] if seg_info['needs_scale'] else X_test.iloc[[i]]
                                seg_pred = seg_info['model'].predict(X_single)[0]

                                if log_transform:
                                    seg_pred = np.expm1(seg_pred)
                                    seg_pred = max(seg_pred, 0)

                                segmented_pred[i] = seg_pred

                            # Calculate combined metrics
                            r2_segmented = r2_score(y_test, segmented_pred)
                            mae_segmented = mean_absolute_error(y_test, segmented_pred)
                            rmse_segmented = np.sqrt(mean_squared_error(y_test, segmented_pred))
                            mask_segmented = y_test != 0
                            mape_segmented = np.mean(np.abs((y_test[mask_segmented] - segmented_pred[mask_segmented]) / y_test[mask_segmented])) * 100
                            pred_acc_segmented = 100 - mape_segmented

                            segmented_name = f"🎯 Segmented ({len(segment_models)} segments)"
                            results[segmented_name] = {
                                'R2': r2_segmented,
                                'Accuracy': r2_segmented * 100,
                                'MAE': mae_segmented,
                                'RMSE': rmse_segmented,
                                'MAPE': mape_segmented,
                                'Prediction_Accuracy': pred_acc_segmented,
                                'predictions': segmented_pred
                            }
                            trained[segmented_name] = {
                                'model': None,
                                'segment_models': segment_models,
                                'segment_stats': segment_stats,
                                'base_model': best_base_model_name,
                                'needs_scale': False,
                                'is_segmented': True,
                                'r2': r2_segmented,
                                'mape': mape_segmented,
                                'prediction_accuracy': pred_acc_segmented,
                                'mae': mae_segmented,
                                'log_transform': log_transform
                            }

                            st.success(f"✅ Segmented Model: R²: {r2_segmented:.2%} | Prediction Accuracy: {pred_acc_segmented:.1f}%")

                            # Show segment breakdown
                            with st.expander("📊 Segment Performance Breakdown"):
                                for seg_name, stats in segment_stats.items():
                                    bounds = segment_bounds[seg_name]
                                    st.write(f"**{seg_name}** (${bounds[0]}-${bounds[1] if bounds[1] != float('inf') else '∞'}):")
                                    st.write(f"  • Samples: {stats['n_train']:,} train, {stats['n_test']:,} test")
                                    st.write(f"  • R²: {stats['R2']:.2%}")
                                    st.write(f"  • Prediction Accuracy: {stats['Prediction_Accuracy']:.1f}%")
                                    st.write(f"  • MAE: ${stats['MAE']:.2f}")
                        else:
                            st.warning("⚠️ Not enough segments trained (need at least 2)")

            st.session_state.model_results = results
            st.session_state.trained_models = trained

            sorted_results = sorted([(k, v) for k, v in results.items() if 'error' not in v], key=lambda x: x[1]['R2'],
                                    reverse=True)
            df_results = pd.DataFrame(
                [{'Model': n, 'R² (%)': f"{r['Accuracy']:.2f}", 'Pred Acc (%)': f"{r['Prediction_Accuracy']:.2f}",
                  'MAE ($)': f"{r['MAE']:.2f}"} for n, r in sorted_results])
            st.dataframe(df_results, use_container_width=True)

            # Show summary of options used
            options_summary = []
            if use_tuned:
                options_summary.append("Tuned Parameters")
            if log_transform:
                options_summary.append("Log Transform")
            if create_ensemble:
                options_summary.append("Ensemble")
            if target_clipping:
                options_summary.append(f"Target Clipping ({clip_percentile}%)")
            if feature_selection:
                options_summary.append("Feature Selection")
            if stacking_ensemble:
                options_summary.append("Stacking")
            if aggressive_tuning:
                options_summary.append("Aggressive Tuning")

            if options_summary:
                st.success(f"✅ Training complete with: {', '.join(options_summary)}")

            if sorted_results[0][1]['R2'] >= 0.8:
                st.balloons()

with tabs[6]:
    st.header("7️⃣ Results")

    if not st.session_state.model_results:
        st.warning("⚠️ Train models first!")
    else:
        results = st.session_state.model_results
        sorted_results = sorted([(k, v) for k, v in results.items() if 'error' not in v], key=lambda x: x[1]['Prediction_Accuracy'],
                                reverse=True)

        # Create comparison dataframe with MAPE and Prediction Accuracy
        df_comp = pd.DataFrame([{
            'Model': n,
            'Prediction Accuracy (%)': round(r['Prediction_Accuracy'], 2),
            'MAPE (%)': round(r['MAPE'], 2),
            'R² (%)': round(r['Accuracy'], 2),
            'MAE ($)': round(r['MAE'], 2),
            'RMSE ($)': round(r['RMSE'], 2)
        } for n, r in sorted_results])
        st.dataframe(df_comp, use_container_width=True)

        # Metrics explanation
        st.markdown("""
        **📊 Metrics Explanation:**
        - **Prediction Accuracy (%)** = 100 - MAPE (your target: 80-100%)
        - **MAPE (%)** = Mean Absolute Percentage Error (lower is better)
        - **R² (%)** = Variance explained by model (higher is better)
        - **MAE ($)** = Average absolute error in dollars
        - **RMSE ($)** = Root mean squared error in dollars
        """)

        # Prediction Accuracy chart (your target metric)
        fig = px.bar(df_comp, x='Model', y='Prediction Accuracy (%)',
                    color='Prediction Accuracy (%)',
                    title="Model Comparison - Prediction Accuracy (Target: 80-100%)",
                    color_continuous_scale='RdYlGn')
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target: 80%")
        fig.add_hline(y=100, line_dash="dash", line_color="blue", annotation_text="Perfect: 100%")
        st.plotly_chart(fig, use_container_width=True)

        # Best model summary
        best_model = sorted_results[0]
        pred_acc = best_model[1]['Prediction_Accuracy']
        mape = best_model[1]['MAPE']

        if pred_acc >= 80:
            st.success(f"🎯 **Best Model: {best_model[0]}** - Prediction Accuracy: **{pred_acc:.2f}%** (MAPE: {mape:.2f}%) ✅ Target achieved!")
            st.balloons()
        elif pred_acc >= 70:
            st.warning(f"📊 **Best Model: {best_model[0]}** - Prediction Accuracy: **{pred_acc:.2f}%** (MAPE: {mape:.2f}%) - Close to target!")
        else:
            st.info(f"📊 **Best Model: {best_model[0]}** - Prediction Accuracy: **{pred_acc:.2f}%** (MAPE: {mape:.2f}%)")

        # =====================================================
        # DIAGNOSTIC EXPORT FOR IMPROVEMENT ANALYSIS
        # =====================================================
        st.markdown("---")
        st.markdown("### 📊 Export Diagnostic Data for Improvement")
        st.markdown("*Export comprehensive data to analyze and improve model performance*")

        if st.button("📥 Generate Diagnostic Report", type="secondary", use_container_width=True):
            diagnostic_data = {}

            # 1. Model Performance Summary
            model_perf = []
            for name, metrics in results.items():
                if 'error' not in metrics:
                    model_perf.append({
                        'Model': name,
                        'R2': metrics['R2'],
                        'Prediction_Accuracy': metrics['Prediction_Accuracy'],
                        'MAPE': metrics['MAPE'],
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE']
                    })
            diagnostic_data['model_performance'] = pd.DataFrame(model_perf)

            # 2. Feature Importances (from best tree-based model)
            feature_importance_data = None
            for model_name in ['XGBoost', 'LightGBM', 'Random Forest', 'CatBoost', 'Gradient Boosting', 'Extra Trees']:
                if model_name in st.session_state.trained_models:
                    model_info = st.session_state.trained_models[model_name]
                    model = model_info['model']
                    if hasattr(model, 'feature_importances_'):
                        features = st.session_state.features
                        importances = model.feature_importances_
                        feature_importance_data = pd.DataFrame({
                            'Feature': features,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        diagnostic_data['feature_importances'] = feature_importance_data
                        break

            # 3. Error Analysis (predictions vs actuals for best model)
            best_model_name = sorted_results[0][0]
            if best_model_name in st.session_state.trained_models and 'predictions' in results[best_model_name]:
                pred = results[best_model_name]['predictions']
                actual = st.session_state.y_test.values

                error_analysis = pd.DataFrame({
                    'Actual_Price': actual,
                    'Predicted_Price': pred,
                    'Error': actual - pred,
                    'Abs_Error': np.abs(actual - pred),
                    'Pct_Error': ((actual - pred) / actual * 100),
                    'Abs_Pct_Error': np.abs((actual - pred) / actual * 100)
                })

                # Add test features for context
                test_features = st.session_state.X_test.copy()
                test_features.reset_index(drop=True, inplace=True)
                error_analysis.reset_index(drop=True, inplace=True)
                error_analysis = pd.concat([error_analysis, test_features], axis=1)

                diagnostic_data['error_analysis'] = error_analysis

                # 4. Error Statistics by Price Range
                error_analysis['Price_Bucket'] = pd.cut(error_analysis['Actual_Price'],
                                                        bins=[0, 50, 100, 150, 200, 300, 500, 1000, float('inf')],
                                                        labels=['$0-50', '$50-100', '$100-150', '$150-200',
                                                                '$200-300', '$300-500', '$500-1000', '$1000+'])
                error_by_bucket = error_analysis.groupby('Price_Bucket').agg({
                    'Abs_Error': ['mean', 'std', 'count'],
                    'Abs_Pct_Error': ['mean', 'std']
                }).round(2)
                error_by_bucket.columns = ['MAE', 'MAE_Std', 'Count', 'MAPE', 'MAPE_Std']
                diagnostic_data['error_by_price_range'] = error_by_bucket.reset_index()

            # 5. Training Settings Used
            training_settings = st.session_state.get('training_settings', {})
            diagnostic_data['training_settings'] = pd.DataFrame([training_settings])

            # 6. Data Summary
            data_summary = {
                'Total_Features': len(st.session_state.features),
                'Train_Size': len(st.session_state.X_train),
                'Test_Size': len(st.session_state.X_test),
                'Target_Mean': st.session_state.y_train.mean(),
                'Target_Std': st.session_state.y_train.std(),
                'Target_Min': st.session_state.y_train.min(),
                'Target_Max': st.session_state.y_train.max(),
                'Target_Median': st.session_state.y_train.median()
            }
            diagnostic_data['data_summary'] = pd.DataFrame([data_summary])

            # 7. Feature Statistics
            feature_stats = st.session_state.X_train.describe().T
            feature_stats['feature'] = feature_stats.index
            diagnostic_data['feature_statistics'] = feature_stats.reset_index(drop=True)

            # 8. Correlation with Target (for improvement ideas)
            correlations = []
            y_train = st.session_state.y_train
            for col in st.session_state.X_train.columns:
                corr = st.session_state.X_train[col].corr(y_train)
                correlations.append({'Feature': col, 'Correlation_With_Price': corr})
            correlation_df = pd.DataFrame(correlations).sort_values('Correlation_With_Price', key=abs, ascending=False)
            diagnostic_data['feature_correlations'] = correlation_df

            # Display summary
            st.success("✅ Diagnostic report generated!")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📋 Report Contents:**")
                for key in diagnostic_data.keys():
                    df = diagnostic_data[key]
                    st.write(f"  • {key}: {len(df)} rows")

            with col2:
                st.markdown("**🎯 Key Improvement Opportunities:**")
                if 'error_by_price_range' in diagnostic_data:
                    worst_bucket = diagnostic_data['error_by_price_range'].sort_values('MAPE', ascending=False).iloc[0]
                    st.write(f"  • Worst performance: {worst_bucket['Price_Bucket']} ({worst_bucket['MAPE']:.1f}% MAPE)")

                if feature_importance_data is not None:
                    top_features = feature_importance_data.head(5)['Feature'].tolist()
                    st.write(f"  • Top features: {', '.join(top_features[:3])}")

                if 'feature_correlations' in diagnostic_data:
                    low_corr = correlation_df[abs(correlation_df['Correlation_With_Price']) < 0.05]
                    st.write(f"  • Low correlation features: {len(low_corr)}")

            # Export options
            st.markdown("---")
            st.markdown("**📥 Download Options:**")

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                # All-in-one Excel export
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for sheet_name, df in diagnostic_data.items():
                        # Truncate sheet name to 31 chars (Excel limit)
                        sheet = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet, index=False)
                buffer.seek(0)
                st.download_button(
                    "📥 Download Full Report (Excel)",
                    buffer,
                    "diagnostic_report.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with export_col2:
                # Error analysis CSV
                if 'error_analysis' in diagnostic_data:
                    st.download_button(
                        "📥 Download Error Analysis (CSV)",
                        diagnostic_data['error_analysis'].to_csv(index=False),
                        "error_analysis.csv",
                        "text/csv"
                    )

            with export_col3:
                # Feature importances CSV
                if 'feature_importances' in diagnostic_data:
                    st.download_button(
                        "📥 Download Feature Importances (CSV)",
                        diagnostic_data['feature_importances'].to_csv(index=False),
                        "feature_importances.csv",
                        "text/csv"
                    )

            # Show preview of key data
            with st.expander("📊 Preview: Error Analysis (Top 20 Worst Predictions)"):
                if 'error_analysis' in diagnostic_data:
                    worst_predictions = diagnostic_data['error_analysis'].nlargest(20, 'Abs_Pct_Error')
                    st.dataframe(worst_predictions[['Actual_Price', 'Predicted_Price', 'Error', 'Abs_Pct_Error']].round(2))

            with st.expander("📊 Preview: Feature Importances (Top 20)"):
                if 'feature_importances' in diagnostic_data:
                    st.dataframe(diagnostic_data['feature_importances'].head(20))

            with st.expander("📊 Preview: Error by Price Range"):
                if 'error_by_price_range' in diagnostic_data:
                    st.dataframe(diagnostic_data['error_by_price_range'])

with tabs[7]:
    st.header("8️⃣ Price Prediction Tool")

    if not st.session_state.trained_models:
        st.warning("⚠️ Train models first!")
    else:
        # Get results and find best models
        results = st.session_state.model_results
        sorted_models = sorted([(k, v) for k, v in results.items() if 'error' not in v],
                               key=lambda x: x[1]['Prediction_Accuracy'], reverse=True)

        if len(sorted_models) < 1:
            st.warning("⚠️ No successful models found!")
        else:
            best_model_name = sorted_models[0][0]

            # Business Context
            st.markdown("""
            ---
            #### 📋 Business Context

            This system helps **Airbnb hosts** determine optimal pricing for their listings.
            Setting the right price is critical for:

            - **Maximizing revenue** while maintaining competitive occupancy rates
            - **Reducing vacancy periods** by avoiding overpricing
            - **Attracting guests** with fair, market-aligned pricing

            ---
            """)

            # Model Performance Section
            st.markdown("### 📊 Model Performance (From Your Training)")

            # Display top 3 models or all if less
            display_models = sorted_models[:min(3, len(sorted_models))]

            cols = st.columns(len(display_models))
            for i, (model_name, metrics) in enumerate(display_models):
                with cols[i]:
                    badge = "🏆 " if i == 0 else ""
                    st.markdown(f"**{badge}{model_name}**")
                    st.metric("R² Score", f"{metrics['R2']:.2%}")
                    st.metric("Prediction Accuracy", f"{metrics['Prediction_Accuracy']:.2f}%")
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    st.metric("MAE", f"${metrics['MAE']:.2f}")

            pred_acc = sorted_models[0][1]['Prediction_Accuracy']
            mape = sorted_models[0][1]['MAPE']
            r2 = sorted_models[0][1]['R2']
            if pred_acc >= 80:
                st.success(f"🎯 **Best Model: {best_model_name}** - R²: **{r2:.2%}** | Prediction Accuracy: **{pred_acc:.2f}%** ✅ Target achieved!")
            else:
                st.info(f"📊 **Best Model: {best_model_name}** - R²: **{r2:.2%}** | Prediction Accuracy: **{pred_acc:.2f}%** (MAPE: {mape:.2f}%)")

            # Metrics explanation
            with st.expander("📖 Understanding the Metrics"):
                st.markdown("""
                | Metric | Description | Interpretation |
                |--------|-------------|----------------|
                | **RMSE** | Root Mean Square Error | Average prediction error in price units ($). Lower = better. |
                | **MAE** | Mean Absolute Error | Average absolute difference between predicted and actual price. Lower = better. |
                | **R²** | Coefficient of Determination | Proportion of variance explained by the model (0-100%). Higher = better. |
                """)

            st.markdown("---")

            # Key Insights
            st.markdown("### 💡 Key Insights from Data Analysis")

            insight_col1, insight_col2 = st.columns(2)
            with insight_col1:
                st.markdown("""
                #### 🏘️ Property Insights
                - **Entire homes/apartments** command the highest prices
                - More **bedrooms** and **bathrooms** increase price significantly
                - **Superhost** status adds a price premium
                - **Amenities count** correlates with higher prices
                """)
            with insight_col2:
                st.markdown("""
                #### 📅 Market Trends
                - Properties with **positive reviews** can charge more
                - **Guest capacity** strongly correlates with price
                - **Instant booking** increases visibility
                - **Location** is a key price factor
                """)

            st.markdown("---")

            # Prediction Interface
            st.markdown("### 🔮 Enter Property Details")

            # Room types and neighbourhoods from encoders
            if st.session_state.le_room is not None:
                room_types = list(st.session_state.le_room.classes_)
            else:
                room_types = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

            if st.session_state.le_neigh is not None:
                neighbourhoods = list(st.session_state.le_neigh.classes_)
            else:
                neighbourhoods = ["Unknown"]

            # Property Details Section
            st.markdown("#### 🏠 Property Details")
            col1, col2, col3 = st.columns(3)

            with col1:
                room_type = st.selectbox("Room Type", options=room_types, key="pred_room")
                bedrooms = st.selectbox("Bedrooms", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1, key="pred_bed")
                bathrooms = st.selectbox("Bathrooms", options=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], index=2, key="pred_bath")

            with col2:
                beds = st.selectbox("Beds", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1, key="pred_beds")
                accommodates = st.selectbox("Accommodates (Guests)", options=list(range(1, 17)), index=1, key="pred_acc")
                minimum_nights = st.selectbox("Minimum Nights", options=[1, 2, 3, 4, 5, 7, 14, 30], index=0, key="pred_min")

            with col3:
                neighbourhood = st.selectbox("Neighbourhood", options=neighbourhoods, key="pred_neigh")
                instant_book = st.selectbox("Instant Bookable", options=["Yes", "No"], index=0, key="pred_instant")
                is_superhost = st.selectbox("Are you a Superhost?", options=["Yes", "No"], index=1, key="pred_super")

            # Location
            st.markdown("#### 📍 Location")
            loc_col1, loc_col2 = st.columns(2)

            # Get center from training data or merged data if available
            default_lat, default_lon = 46.0878, -64.7782  # Fallback defaults
            if st.session_state.merged_data is not None and 'latitude' in st.session_state.merged_data.columns:
                default_lat = st.session_state.merged_data['latitude'].median()
                default_lon = st.session_state.merged_data['longitude'].median()
            elif st.session_state.X_train is not None and 'latitude' in st.session_state.X_train.columns and 'longitude' in st.session_state.X_train.columns:
                default_lat = st.session_state.X_train['latitude'].median()
                default_lon = st.session_state.X_train['longitude'].median()

            with loc_col1:
                latitude = st.number_input("Latitude", value=float(default_lat), format="%.4f", key="pred_lat")
            with loc_col2:
                longitude = st.number_input("Longitude", value=float(default_lon), format="%.4f", key="pred_lon")

            # Host & Listing Details
            st.markdown("#### 👤 Host & Listing Details")
            detail_col1, detail_col2, detail_col3 = st.columns(3)

            with detail_col1:
                host_verified = st.selectbox("Host Identity Verified", options=["Yes", "No"], index=0, key="pred_verified")
                host_listings_count = st.number_input("Total Listings by Host", min_value=1, max_value=100, value=1, key="pred_listings")

            with detail_col2:
                amenities_count = st.slider("Number of Amenities", min_value=0, max_value=100, value=20, key="pred_amen")
                description_length = st.slider("Description Length (chars)", min_value=0, max_value=2000, value=500, key="pred_desc")

            with detail_col3:
                has_luxury = st.selectbox("Luxury Keywords in Title?", options=["Yes", "No"], index=1, key="pred_lux")

            # Review Scores
            st.markdown("#### ⭐ Review Scores (0-5 scale)")
            review_col1, review_col2, review_col3, review_col4 = st.columns(4)

            with review_col1:
                review_rating = st.slider("Overall Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1, key="pred_rating")
            with review_col2:
                review_cleanliness = st.slider("Cleanliness", min_value=0.0, max_value=5.0, value=4.5, step=0.1, key="pred_clean")
            with review_col3:
                review_location = st.slider("Location", min_value=0.0, max_value=5.0, value=4.5, step=0.1, key="pred_loc")
            with review_col4:
                review_value = st.slider("Value", min_value=0.0, max_value=5.0, value=4.5, step=0.1, key="pred_val")

            review_col5, review_col6 = st.columns(2)
            with review_col5:
                number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10, key="pred_numrev")
            with review_col6:
                reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, max_value=30.0, value=1.0, step=0.1, key="pred_rpm")

            # Month Selection for Seasonal Pricing
            st.markdown("#### 📅 Prediction Month")

            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']

            # Check if monthly stats are available
            has_monthly_data = st.session_state.get('monthly_price_stats') is not None

            month_col1, month_col2 = st.columns(2)
            with month_col1:
                selected_month = st.selectbox(
                    "Select Month for Price Prediction",
                    options=month_names,
                    index=datetime.now().month - 1,  # Default to current month
                    key="pred_month",
                    help="Prices vary by season. Select the month you want to predict for."
                )

            with month_col2:
                if has_monthly_data:
                    monthly_stats = st.session_state.monthly_price_stats
                    month_num = month_names.index(selected_month) + 1
                    month_data = monthly_stats[monthly_stats['month'] == month_num]

                    if len(month_data) > 0:
                        price_index = month_data['price_index'].values[0]
                        avg_price = month_data['avg_price'].values[0]
                        availability = month_data['availability'].values[0]

                        # Show seasonal indicator
                        if price_index > 1.05:
                            season_icon = "🔥"
                            season_label = "Peak Season"
                        elif price_index < 0.95:
                            season_icon = "❄️"
                            season_label = "Low Season"
                        else:
                            season_icon = "📊"
                            season_label = "Normal Season"

                        st.metric(
                            f"{season_icon} {season_label}",
                            f"{(price_index-1)*100:+.1f}%",
                            help=f"Avg price: ${avg_price:.2f}, Availability: {availability:.1%}"
                        )
                    else:
                        st.info("No data for this month")
                else:
                    st.warning("⚠️ Process calendar data first for monthly adjustments")

            # Show monthly price chart if available
            if has_monthly_data:
                with st.expander("📊 View Monthly Price Trends"):
                    monthly_stats = st.session_state.monthly_price_stats

                    # Use training data mean for consistency with predictions
                    if st.session_state.y_train is not None:
                        training_mean = st.session_state.y_train.mean()
                    else:
                        training_mean = monthly_stats['avg_price'].mean()

                    # Scale monthly prices to match training data distribution
                    monthly_stats_display = monthly_stats.copy()
                    monthly_stats_display['display_price'] = training_mean * monthly_stats_display['price_index']

                    fig = px.bar(monthly_stats_display, x='month_name', y='display_price',
                                title="Market Price Trends by Month (Scaled to Training Data)",
                                color='price_index',
                                color_continuous_scale='RdYlGn_r')
                    fig.add_hline(y=training_mean, line_dash="dash", line_color="red",
                                 annotation_text=f"Training Avg: ${training_mean:.0f}")
                    fig.update_layout(height=300, xaxis_title="Month", yaxis_title="Average Price ($)")

                    # Highlight selected month
                    month_num = month_names.index(selected_month) + 1
                    st.plotly_chart(fig, use_container_width=True)

                    # Peak/Low info
                    peak_month = monthly_stats.loc[monthly_stats['price_index'].idxmax(), 'month_name']
                    low_month = monthly_stats.loc[monthly_stats['price_index'].idxmin(), 'month_name']
                    st.caption(f"🔥 Peak: {peak_month} | ❄️ Low: {low_month}")

            # Model Selection
            st.markdown("#### 🤖 Model Selection")
            model_options = [m[0] for m in sorted_models]
            selected_models_pred = st.multiselect("Select Models for Prediction:",
                                                   options=model_options,
                                                   default=model_options[:min(2, len(model_options))],
                                                   key="pred_models")

            # Predict Button
            if st.button("🎯 Predict Price", type="primary", use_container_width=True, key="pred_btn"):
                if not selected_models_pred:
                    st.warning("Please select at least one model!")
                else:
                    try:
                        # Encode categorical variables
                        if st.session_state.le_room is not None and room_type in st.session_state.le_room.classes_:
                            room_type_enc = st.session_state.le_room.transform([room_type])[0]
                        else:
                            room_type_enc = 0

                        if st.session_state.le_neigh is not None and neighbourhood in st.session_state.le_neigh.classes_:
                            neighbourhood_enc = st.session_state.le_neigh.transform([neighbourhood])[0]
                        else:
                            neighbourhood_enc = 0

                        # Calculate derived features
                        dist_center = np.sqrt((latitude - default_lat) ** 2 + (longitude - default_lon) ** 2)
                        beds_per_bedroom = beds / max(bedrooms, 1)
                        baths_per_bedroom = bathrooms / max(bedrooms, 1)
                        capacity_score = accommodates * max(bedrooms, 1) * max(beds, 1)
                        space_score = accommodates + bedrooms * 2 + beds + bathrooms * 1.5
                        review_composite = (review_rating + review_cleanliness + review_location + review_value) / 4

                        # Binary conversions
                        is_superhost_val = 1 if is_superhost == "Yes" else 0
                        host_verified_val = 1 if host_verified == "Yes" else 0
                        instant_book_val = 1 if instant_book == "Yes" else 0
                        has_luxury_val = 1 if has_luxury == "Yes" else 0

                        # === LOOKUP ACTUAL VALUES FROM TRAINING DATA ===
                        # Get merged data for lookups
                        merged = st.session_state.merged_data

                        # --- Neighbourhood Statistics (from actual data) ---
                        neigh_stats = {
                            'neigh_centroid_lon': longitude,
                            'neigh_centroid_lat': latitude,
                            'neigh_area': 0.01,
                            'neigh_perimeter': 0.1,
                            'neigh_group_enc': 0,
                            'neigh_listing_count': 50,
                            'neigh_density': 5000,
                            'neigh_avg_rating': 4.5,
                            'neigh_avg_accommodates': 4,
                            'neigh_avg_bedrooms': 2,
                            'neigh_avg_reviews': 20,
                            'dist_to_neigh_center': 0.001
                        }

                        if merged is not None and 'neighbourhood_enc' in merged.columns:
                            neigh_data = merged[merged['neighbourhood_enc'] == neighbourhood_enc]
                            if len(neigh_data) > 0:
                                # Use actual neighbourhood statistics
                                neigh_stats['neigh_listing_count'] = len(neigh_data)
                                if 'latitude' in neigh_data.columns and 'longitude' in neigh_data.columns:
                                    neigh_stats['neigh_centroid_lat'] = neigh_data['latitude'].mean()
                                    neigh_stats['neigh_centroid_lon'] = neigh_data['longitude'].mean()
                                    # Calculate distance to neighborhood center
                                    neigh_stats['dist_to_neigh_center'] = np.sqrt(
                                        (latitude - neigh_stats['neigh_centroid_lat'])**2 +
                                        (longitude - neigh_stats['neigh_centroid_lon'])**2
                                    )
                                if 'review_scores_rating' in neigh_data.columns:
                                    neigh_stats['neigh_avg_rating'] = neigh_data['review_scores_rating'].mean()
                                if 'accommodates' in neigh_data.columns:
                                    neigh_stats['neigh_avg_accommodates'] = neigh_data['accommodates'].mean()
                                if 'bedrooms' in neigh_data.columns:
                                    neigh_stats['neigh_avg_bedrooms'] = neigh_data['bedrooms'].mean()
                                if 'number_of_reviews' in neigh_data.columns:
                                    neigh_stats['neigh_avg_reviews'] = neigh_data['number_of_reviews'].mean()
                                # Calculate density (listings per unit area approximation)
                                if 'latitude' in neigh_data.columns and 'longitude' in neigh_data.columns:
                                    lat_range = neigh_data['latitude'].max() - neigh_data['latitude'].min()
                                    lon_range = neigh_data['longitude'].max() - neigh_data['longitude'].min()
                                    area = max((lat_range * lon_range), 0.0001)
                                    neigh_stats['neigh_area'] = area
                                    neigh_stats['neigh_density'] = len(neigh_data) / area
                                # Lookup actual values from training data if available
                                for col in ['neigh_centroid_lon', 'neigh_centroid_lat', 'neigh_area',
                                           'neigh_perimeter', 'neigh_group_enc', 'neigh_listing_count',
                                           'neigh_density', 'neigh_avg_rating', 'neigh_avg_accommodates',
                                           'neigh_avg_bedrooms', 'neigh_avg_reviews', 'dist_to_neigh_center']:
                                    if col in neigh_data.columns:
                                        neigh_stats[col] = neigh_data[col].median()

                        # --- Calendar Price Statistics (from similar properties) ---
                        cal_stats = {
                            'cal_price_mean': 100,
                            'cal_price_std': 25,
                            'cal_price_min': 50,
                            'cal_price_max': 200,
                            'cal_price_median': 95,
                            'cal_price_q25': 75,
                            'cal_price_q75': 130,
                            'cal_avail_rate': 0.7,
                            'price_range': 150,
                            'price_iqr': 55,
                            'price_volatility': 0.25,
                            'dynamic_pricing': 0,
                            'high_demand': 0
                        }

                        if merged is not None:
                            # Find similar properties (same room type, similar size)
                            similar_props = merged  # Start with all data as fallback

                            # Try to filter by room type if column exists
                            if 'room_type_enc' in merged.columns:
                                similar_mask = (merged['room_type_enc'] == room_type_enc)
                                if 'bedrooms' in merged.columns:
                                    similar_mask &= (merged['bedrooms'] >= max(0, bedrooms - 1)) & (merged['bedrooms'] <= bedrooms + 1)
                                similar_props = merged[similar_mask]

                                # If not enough similar, fall back to same neighbourhood
                                if len(similar_props) < 5 and 'neighbourhood_enc' in merged.columns:
                                    similar_props = merged[merged['neighbourhood_enc'] == neighbourhood_enc]

                                # If still not enough, use all data
                                if len(similar_props) < 5:
                                    similar_props = merged

                            # Lookup actual calendar statistics
                            for col in cal_stats.keys():
                                if col in similar_props.columns:
                                    val = similar_props[col].median()
                                    if pd.notna(val):
                                        cal_stats[col] = val

                        # --- Sentiment/Review Statistics (from similar properties) ---
                        sentiment_stats = {
                            'review_count': number_of_reviews,
                            'avg_comment_len': 150,
                            'avg_comment_words': 30,
                            'max_comment_words': 100,
                            'total_positive': int(number_of_reviews * 0.8),
                            'total_negative': int(number_of_reviews * 0.1),
                            'total_sentiment': int(number_of_reviews * 0.7),
                            'avg_sentiment': 0.4,
                            'sentiment_ratio': 8.0,
                            'days_since_review': 30,
                            'has_recent_review': 1 if number_of_reviews > 0 else 0
                        }

                        if merged is not None:
                            # Lookup actual sentiment statistics from training data
                            for col in sentiment_stats.keys():
                                if col in merged.columns:
                                    val = merged[col].median()
                                    if pd.notna(val):
                                        sentiment_stats[col] = val

                            # Scale sentiment by user's review rating vs average
                            avg_rating_in_data = merged['review_scores_rating'].mean() if 'review_scores_rating' in merged.columns else 4.5
                            rating_ratio = review_rating / max(avg_rating_in_data, 1)
                            sentiment_stats['avg_sentiment'] = sentiment_stats['avg_sentiment'] * rating_ratio
                            sentiment_stats['total_positive'] = int(sentiment_stats['total_positive'] * rating_ratio)

                        # --- Host Statistics (from training data) ---
                        host_response = 0.9
                        host_acceptance = 0.9
                        if merged is not None:
                            if 'host_response_rate' in merged.columns:
                                # Handle both numeric and string percentage formats
                                hr_series = merged['host_response_rate'].copy()
                                # Always convert to string first, then clean and convert to numeric
                                hr_series = hr_series.astype(str).str.replace('%', '', regex=False).str.replace('nan', '', regex=False)
                                hr_series = pd.to_numeric(hr_series, errors='coerce')
                                # Convert from 0-100 scale to 0-1 if needed
                                hr_median = hr_series.median()
                                if pd.notna(hr_median) and hr_median > 1:
                                    hr_series = hr_series / 100
                                val = hr_series.median()
                                if pd.notna(val):
                                    host_response = val
                            if 'host_acceptance_rate' in merged.columns:
                                # Handle both numeric and string percentage formats
                                ha_series = merged['host_acceptance_rate'].copy()
                                # Always convert to string first, then clean and convert to numeric
                                ha_series = ha_series.astype(str).str.replace('%', '', regex=False).str.replace('nan', '', regex=False)
                                ha_series = pd.to_numeric(ha_series, errors='coerce')
                                # Convert from 0-100 scale to 0-1 if needed
                                ha_median = ha_series.median()
                                if pd.notna(ha_median) and ha_median > 1:
                                    ha_series = ha_series / 100
                                val = ha_series.median()
                                if pd.notna(val):
                                    host_acceptance = val

                        # Create feature dictionary with ACTUAL looked-up values
                        feature_values = {
                            'accommodates': accommodates,
                            'bedrooms': bedrooms,
                            'beds': beds,
                            'bathrooms': bathrooms,
                            'latitude': latitude,
                            'longitude': longitude,
                            'neighbourhood_enc': neighbourhood_enc,
                            'dist_center': dist_center,
                            'room_type_enc': room_type_enc,
                            'property_type_enc': 0,
                            'is_superhost': is_superhost_val,
                            'host_verified': host_verified_val,
                            'has_profile_pic': 1,
                            'host_listings_count': host_listings_count,
                            'host_response_rate': host_response,
                            'host_acceptance_rate': host_acceptance,
                            'instant_book': instant_book_val,
                            'minimum_nights': minimum_nights,
                            'amenities_count': amenities_count,
                            'desc_length': description_length,
                            'name_length': 50,
                            'has_luxury': has_luxury_val,
                            'premium_amenities': 2,
                            'review_scores_rating': review_rating,
                            'review_scores_cleanliness': review_cleanliness,
                            'review_scores_location': review_location,
                            'review_scores_value': review_value,
                            'review_composite': review_composite,
                            'review_min': min(review_rating, review_cleanliness, review_location, review_value),
                            'review_std': np.std([review_rating, review_cleanliness, review_location, review_value]),
                            'number_of_reviews': number_of_reviews,
                            'reviews_per_month': reviews_per_month,
                            'beds_per_bedroom': beds_per_bedroom,
                            'baths_per_bedroom': baths_per_bedroom,
                            'capacity_score': capacity_score,
                            'space_score': space_score,
                            # Calendar stats from actual data
                            'cal_price_mean': cal_stats['cal_price_mean'],
                            'cal_price_std': cal_stats['cal_price_std'],
                            'cal_price_min': cal_stats['cal_price_min'],
                            'cal_price_max': cal_stats['cal_price_max'],
                            'cal_price_median': cal_stats['cal_price_median'],
                            'cal_price_q25': cal_stats['cal_price_q25'],
                            'cal_price_q75': cal_stats['cal_price_q75'],
                            'cal_avail_rate': cal_stats['cal_avail_rate'],
                            'price_range': cal_stats['price_range'],
                            'price_iqr': cal_stats['price_iqr'],
                            'price_volatility': cal_stats['price_volatility'],
                            'dynamic_pricing': cal_stats['dynamic_pricing'],
                            'high_demand': cal_stats['high_demand'],
                            # Sentiment stats from actual data
                            'review_count': sentiment_stats['review_count'],
                            'avg_comment_len': sentiment_stats['avg_comment_len'],
                            'avg_comment_words': sentiment_stats['avg_comment_words'],
                            'max_comment_words': sentiment_stats['max_comment_words'],
                            'total_positive': sentiment_stats['total_positive'],
                            'total_negative': sentiment_stats['total_negative'],
                            'total_sentiment': sentiment_stats['total_sentiment'],
                            'avg_sentiment': sentiment_stats['avg_sentiment'],
                            'sentiment_ratio': sentiment_stats['sentiment_ratio'],
                            'days_since_review': sentiment_stats['days_since_review'],
                            'has_recent_review': sentiment_stats['has_recent_review'],
                            # Neighbourhood stats from actual data
                            'neigh_centroid_lon': neigh_stats['neigh_centroid_lon'],
                            'neigh_centroid_lat': neigh_stats['neigh_centroid_lat'],
                            'neigh_area': neigh_stats['neigh_area'],
                            'neigh_perimeter': neigh_stats['neigh_perimeter'],
                            'neigh_group_enc': neigh_stats['neigh_group_enc'],
                            'neigh_listing_count': neigh_stats['neigh_listing_count'],
                            'neigh_density': neigh_stats['neigh_density'],
                            'neigh_avg_rating': neigh_stats['neigh_avg_rating'],
                            'neigh_avg_accommodates': neigh_stats['neigh_avg_accommodates'],
                            'neigh_avg_bedrooms': neigh_stats['neigh_avg_bedrooms'],
                            'neigh_avg_reviews': neigh_stats['neigh_avg_reviews'],
                            'dist_to_neigh_center': neigh_stats['dist_to_neigh_center']
                        }

                        # Create DataFrame with features
                        features = st.session_state.features

                        # Validate features list exists
                        if features is None:
                            st.error("❌ Features not found. Please re-run Feature Engineering and Training.")
                            st.stop()

                        # Ensure all expected features have values (use 0 for missing)
                        user_df = pd.DataFrame([{f: feature_values.get(f, 0) for f in features}])

                        # Ensure column order matches exactly what models expect
                        user_df = user_df[features]

                        # Make predictions with selected models
                        predictions = {}
                        for model_name in selected_models_pred:
                            model_info = st.session_state.trained_models[model_name]
                            if model_info['needs_scale']:
                                user_scaled = pd.DataFrame(
                                    st.session_state.scaler.transform(user_df),
                                    columns=features
                                )
                                pred = model_info['model'].predict(user_scaled)[0]
                            else:
                                pred = model_info['model'].predict(user_df)[0]

                            # Reverse log transform if it was used during training
                            if model_info.get('log_transform', False):
                                pred = np.expm1(pred)  # exp(y) - 1 to reverse log1p

                            predictions[model_name] = max(pred, 10)  # Ensure positive

                        # Calculate average
                        avg_pred = np.mean(list(predictions.values()))

                        # Apply monthly seasonal adjustment
                        monthly_adjustment = 1.0
                        monthly_adjustment_pct = 0.0
                        if has_monthly_data:
                            monthly_stats = st.session_state.monthly_price_stats
                            month_num = month_names.index(selected_month) + 1
                            month_data = monthly_stats[monthly_stats['month'] == month_num]
                            if len(month_data) > 0:
                                monthly_adjustment = month_data['price_index'].values[0]
                                monthly_adjustment_pct = (monthly_adjustment - 1) * 100

                        # Apply adjustment to all predictions
                        adjusted_predictions = {k: v * monthly_adjustment for k, v in predictions.items()}
                        avg_pred_adjusted = avg_pred * monthly_adjustment

                        # Display Results
                        st.markdown("---")
                        st.markdown(f"### 💰 Predicted Nightly Prices for {selected_month}")

                        # Show seasonal adjustment info if applicable
                        if has_monthly_data and abs(monthly_adjustment_pct) > 0.5:
                            adj_direction = "higher" if monthly_adjustment_pct > 0 else "lower"
                            st.info(f"📅 **Seasonal Adjustment for {selected_month}:** Prices are typically {abs(monthly_adjustment_pct):.1f}% {adj_direction} than average based on historical data.")

                        result_cols = st.columns(len(adjusted_predictions) + 1)
                        for i, (model_name, pred) in enumerate(adjusted_predictions.items()):
                            with result_cols[i]:
                                r2 = results[model_name]['R2']
                                badge = "🏆 " if model_name == best_model_name else ""
                                st.metric(label=f"{badge}{model_name}", value=f"${pred:,.2f}",
                                          help=f"R² = {r2:.2%}")

                        with result_cols[-1]:
                            st.metric(label="📊 Average", value=f"${avg_pred_adjusted:,.2f}")

                        # Recommendation
                        seasonal_note = f" (adjusted for {selected_month})" if has_monthly_data and abs(monthly_adjustment_pct) > 0.5 else ""
                        st.info(f"""
                        **💡 Pricing Recommendation for {selected_month}**

                        Based on your property characteristics{seasonal_note}, we recommend pricing your listing at approximately **${avg_pred_adjusted:,.2f}** per night.

                        **Suggested Price Range for {selected_month}:**
                        - 💰 **Budget-friendly** (faster bookings): ${avg_pred_adjusted*0.85:,.2f} - ${avg_pred_adjusted*0.95:,.2f}
                        - ⭐ **Competitive** (balanced): ${avg_pred_adjusted*0.95:,.2f} - ${avg_pred_adjusted*1.05:,.2f}
                        - 💎 **Premium** (high demand periods): ${avg_pred_adjusted*1.1:,.2f} - ${avg_pred_adjusted*1.25:,.2f}
                        """)

                        # Yearly Price Prediction Chart (all 12 months)
                        with st.expander("📅 Yearly Price Predictions (All 12 Months)", expanded=True):
                            yearly_predictions = []
                            for m_idx, m_name in enumerate(month_names):
                                m_num = m_idx + 1
                                m_adj = 1.0
                                if has_monthly_data:
                                    m_data = monthly_stats[monthly_stats['month'] == m_num]
                                    if len(m_data) > 0:
                                        m_adj = m_data['price_index'].values[0]
                                    else:
                                        # Interpolate: use average of available months' price_index
                                        m_adj = monthly_stats['price_index'].mean() if len(monthly_stats) > 0 else 1.0
                                yearly_predictions.append({
                                    'month': m_name,
                                    'month_num': m_num,
                                    'predicted_price': avg_pred * m_adj,
                                    'adjustment': m_adj,
                                    'has_data': has_monthly_data and m_num in monthly_stats['month'].values
                                })

                            yearly_df = pd.DataFrame(yearly_predictions)

                            # Create chart
                            fig_yearly = px.bar(yearly_df, x='month', y='predicted_price',
                                               title=f"Predicted Price by Month (Best Model: {best_model_name})",
                                               color='adjustment',
                                               color_continuous_scale='RdYlGn_r',
                                               hover_data=['adjustment', 'has_data'])
                            fig_yearly.add_hline(y=avg_pred, line_dash="dash", line_color="blue",
                                                annotation_text=f"Base: ${avg_pred:.0f}")
                            fig_yearly.update_layout(height=350, xaxis_title="Month", yaxis_title="Predicted Price ($)")
                            st.plotly_chart(fig_yearly, use_container_width=True)

                            # Show data availability warning
                            months_with_data = yearly_df[yearly_df['has_data']]['month'].tolist()
                            months_without_data = yearly_df[~yearly_df['has_data']]['month'].tolist()
                            if months_without_data:
                                st.warning(f"⚠️ No historical data for: {', '.join(months_without_data)}. Using average adjustment.")
                            if months_with_data:
                                st.caption(f"📊 Historical data available for: {', '.join(months_with_data)}")

                            # Price summary table
                            col_t1, col_t2 = st.columns(2)
                            with col_t1:
                                st.write("**Price Range:**")
                                st.write(f"- Min: ${yearly_df['predicted_price'].min():,.2f} ({yearly_df.loc[yearly_df['predicted_price'].idxmin(), 'month']})")
                                st.write(f"- Max: ${yearly_df['predicted_price'].max():,.2f} ({yearly_df.loc[yearly_df['predicted_price'].idxmax(), 'month']})")
                            with col_t2:
                                st.write("**Seasonal Variation:**")
                                variation = (yearly_df['predicted_price'].max() - yearly_df['predicted_price'].min()) / avg_pred * 100
                                st.write(f"- Variation: {variation:.1f}%")
                                st.write(f"- Avg Price: ${yearly_df['predicted_price'].mean():,.2f}")

                        # Price factors breakdown
                        with st.expander("📈 What Factors Influenced This Price?"):
                            factors = []

                            # Seasonal factor
                            if has_monthly_data and abs(monthly_adjustment_pct) > 0.5:
                                if monthly_adjustment_pct > 0:
                                    factors.append(f"📅 **{selected_month} (Peak Season)** - Prices {monthly_adjustment_pct:.1f}% higher than average")
                                else:
                                    factors.append(f"📅 **{selected_month} (Low Season)** - Prices {abs(monthly_adjustment_pct):.1f}% lower than average")

                            if room_type == "Entire home/apt":
                                factors.append("✅ **Entire home/apt** - Highest demand category")
                            if is_superhost_val:
                                factors.append("✅ **Superhost status** - Adds trust premium")
                            if bedrooms >= 2:
                                factors.append(f"✅ **{bedrooms} bedrooms** - Good capacity")
                            if review_rating >= 4.5:
                                factors.append(f"✅ **{review_rating} rating** - Excellent reviews")
                            if instant_book_val:
                                factors.append("✅ **Instant booking** - Increases visibility")
                            if amenities_count >= 20:
                                factors.append(f"✅ **{amenities_count} amenities** - Well-equipped")
                            if has_luxury_val:
                                factors.append("✅ **Luxury keywords** - Premium positioning")

                            if review_rating < 4.0:
                                factors.append(f"⚠️ **{review_rating} rating** - Below average may reduce bookings")
                            if number_of_reviews < 5:
                                factors.append(f"⚠️ **{number_of_reviews} reviews** - New listings may need lower prices initially")
                            if minimum_nights > 3:
                                factors.append(f"⚠️ **{minimum_nights} min nights** - May limit short-stay guests")

                            for factor in factors:
                                st.markdown(factor)

                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                        st.info("Please check that all inputs are valid and try again.")

            # Test Set Predictions Section
            st.markdown("---")
            st.markdown("### 📊 Test Set Predictions")

            # Dynamic model selection for test predictions
            available_models = list(st.session_state.trained_models.keys())

            # Sort models by R2 score (best first) and add performance indicator
            model_options = []
            for model_name in available_models:
                model_info = st.session_state.trained_models[model_name]
                r2 = model_info.get('r2', 0)
                pred_acc = model_info.get('prediction_accuracy', 0)
                if model_name == best_model_name:
                    model_options.append(f"🏆 {model_name} (Best - R²: {r2:.2%}, Acc: {pred_acc:.1f}%)")
                else:
                    model_options.append(f"{model_name} (R²: {r2:.2%}, Acc: {pred_acc:.1f}%)")

            col_select, col_btn = st.columns([3, 1])
            with col_select:
                selected_model_display = st.selectbox(
                    "Select Model for Test Predictions",
                    options=model_options,
                    index=0,  # Default to best model
                    key="test_pred_model_select"
                )

            # Extract actual model name from display string
            selected_model_name = selected_model_display.replace("🏆 ", "").split(" (")[0]

            with col_btn:
                st.write("")  # Spacer for alignment
                show_predictions = st.button("🔍 Show Predictions", key="test_pred_btn")

            if show_predictions:
                model_info = st.session_state.trained_models[selected_model_name]
                X_te = st.session_state.X_test_scaled if model_info['needs_scale'] else st.session_state.X_test
                pred = model_info['model'].predict(X_te)

                # Reverse log transform if it was used during training
                if model_info.get('log_transform', False):
                    pred = np.expm1(pred)  # exp(y) - 1 to reverse log1p
                    pred = np.maximum(pred, 0)  # Ensure non-negative

                df_pred = pd.DataFrame({
                    'Actual': st.session_state.y_test.values,
                    'Predicted': pred.round(2),
                    'Error': (st.session_state.y_test.values - pred).round(2),
                    'Error %': ((st.session_state.y_test.values - pred) / st.session_state.y_test.values * 100).round(1)
                })

                # Show model metrics for selected model
                r2 = model_info.get('r2', 0)
                pred_acc = model_info.get('prediction_accuracy', 0)
                mape = model_info.get('mape', 0)
                mae = model_info.get('mae', 0)

                st.info(f"**{selected_model_name}** | R²: {r2:.2%} | Prediction Accuracy: {pred_acc:.1f}% | MAPE: {mape:.1f}% | MAE: ${mae:.2f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(df_pred.head(50), use_container_width=True)
                with col2:
                    fig = px.scatter(df_pred.head(200), x='Actual', y='Predicted',
                                     title=f"Actual vs Predicted ({selected_model_name})")
                    fig.add_trace(go.Scatter(x=[df_pred['Actual'].min(), df_pred['Actual'].max()],
                                             y=[df_pred['Actual'].min(), df_pred['Actual'].max()],
                                             mode='lines', name='Perfect Prediction',
                                             line=dict(dash='dash', color='red')))
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button("📥 Download Predictions", df_pred.to_csv(index=False), f"predictions_{selected_model_name}.csv")

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:gray'>🏠 Airbnb ML Studio | {len(ALL_MODELS)} Models | Advanced Cleaning & Logging</div>",
    unsafe_allow_html=True)