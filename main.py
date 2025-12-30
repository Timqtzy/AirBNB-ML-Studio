import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import zipfile
import tempfile
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

# All available models
ALL_MODELS = {
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

if HAS_XGB:
    ALL_MODELS["XGBoost"] = {
        "model": xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42, n_jobs=-1), "category": "Boosting",
        "scale": False}
if HAS_LGB:
    ALL_MODELS["LightGBM"] = {
        "model": lgb.LGBMRegressor(n_estimators=500, max_depth=12, learning_rate=0.05, num_leaves=50, subsample=0.8,
                                   random_state=42, n_jobs=-1, verbose=-1), "category": "Boosting", "scale": False}
if HAS_CAT:
    ALL_MODELS["CatBoost"] = {
        "model": cb.CatBoostRegressor(iterations=500, depth=10, learning_rate=0.05, random_state=42, verbose=0),
        "category": "Boosting", "scale": False}


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
            'le_room', 'le_neigh', 'le_prop']:
    if key not in st.session_state:
        st.session_state[key] = None

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

    uploaded_zips = st.file_uploader("📁 Upload ZIP File(s)", type=['zip'], accept_multiple_files=True)

    if uploaded_zips is not None and len(uploaded_zips) > 0:
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

            # Combine all dataframes
            if all_listings:
                combined_listings = pd.concat(all_listings, ignore_index=True)
                # Remove duplicate listings by ID, keep the most recent (last occurrence)
                if 'id' in combined_listings.columns:
                    before_dedup = len(combined_listings)
                    combined_listings = combined_listings.drop_duplicates(subset=['id'], keep='last')
                    dedup_count = before_dedup - len(combined_listings)
                    if dedup_count > 0:
                        st.info(f"ℹ️ Removed {dedup_count:,} duplicate listings (by ID) from combined data")
                st.session_state.listings = combined_listings

            if all_calendar:
                combined_calendar = pd.concat(all_calendar, ignore_index=True)
                # For calendar, remove duplicates by listing_id + date combination
                if 'listing_id' in combined_calendar.columns and 'date' in combined_calendar.columns:
                    before_dedup = len(combined_calendar)
                    combined_calendar = combined_calendar.drop_duplicates(subset=['listing_id', 'date'], keep='last')
                    dedup_count = before_dedup - len(combined_calendar)
                    if dedup_count > 0:
                        st.info(f"ℹ️ Removed {dedup_count:,} duplicate calendar entries (by listing_id + date)")
                st.session_state.calendar = combined_calendar

            if all_reviews:
                combined_reviews = pd.concat(all_reviews, ignore_index=True)
                # Remove duplicate reviews by ID
                if 'id' in combined_reviews.columns:
                    before_dedup = len(combined_reviews)
                    combined_reviews = combined_reviews.drop_duplicates(subset=['id'], keep='last')
                    dedup_count = before_dedup - len(combined_reviews)
                    if dedup_count > 0:
                        st.info(f"ℹ️ Removed {dedup_count:,} duplicate reviews (by ID)")
                st.session_state.reviews = combined_reviews

            if all_neighbourhoods:
                combined_neighbourhoods = pd.concat(all_neighbourhoods, ignore_index=True)
                # Remove duplicate neighbourhoods by name
                if 'neighbourhood' in combined_neighbourhoods.columns:
                    combined_neighbourhoods = combined_neighbourhoods.drop_duplicates(subset=['neighbourhood'], keep='last')
                st.session_state.neighbourhoods = combined_neighbourhoods

        st.success(f"✅ Files extracted successfully from {len(uploaded_zips)} dataset(s)!")

        # Show dataset sources
        if len(dataset_sources) > 1:
            st.info(f"📊 **Combined Datasets:** {', '.join(dataset_sources)}")

        # Show files found per dataset
        st.subheader("📋 Files Loaded")
        for dtype, fname, count, source in files_found:
            source_label = f" [{source}]" if len(dataset_sources) > 1 else ""
            st.write(f"  ✅ **{dtype}**: `{fname}` ({count:,} rows){source_label}")

        # Metrics (combined totals)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.session_state.listings is not None:
                st.metric("🏠 Listings", f"{len(st.session_state.listings):,}")
                source_count = st.session_state.listings['_source_dataset'].nunique() if '_source_dataset' in st.session_state.listings.columns else 1
                st.caption(f"{len(st.session_state.listings.columns)} cols | {source_count} source(s)")
            else:
                st.error("❌ Missing")
        with col2:
            if st.session_state.calendar is not None:
                st.metric("📅 Calendar", f"{len(st.session_state.calendar):,}")
                source_count = st.session_state.calendar['_source_dataset'].nunique() if '_source_dataset' in st.session_state.calendar.columns else 1
                st.caption(f"{len(st.session_state.calendar.columns)} cols | {source_count} source(s)")
            else:
                st.warning("⚠️ Not found")
        with col3:
            if st.session_state.reviews is not None:
                st.metric("⭐ Reviews", f"{len(st.session_state.reviews):,}")
                source_count = st.session_state.reviews['_source_dataset'].nunique() if '_source_dataset' in st.session_state.reviews.columns else 1
                st.caption(f"{len(st.session_state.reviews.columns)} cols | {source_count} source(s)")
            else:
                st.warning("⚠️ Not found")
        with col4:
            if st.session_state.neighbourhoods is not None:
                st.metric("📍 Neighbourhoods", f"{len(st.session_state.neighbourhoods):,}")
                source_count = st.session_state.neighbourhoods['_source_dataset'].nunique() if '_source_dataset' in st.session_state.neighbourhoods.columns else 1
                st.caption(f"{len(st.session_state.neighbourhoods.columns)} cols | {source_count} source(s)")
            else:
                st.warning("⚠️ Not found")

        # Data preview
        if st.checkbox("📋 Preview Raw Data"):
            preview_choice = st.selectbox("Select dataset:",
                                          [k for k in ['listings', 'calendar', 'reviews', 'neighbourhoods']
                                           if st.session_state.get(k) is not None])
            if preview_choice:
                data = st.session_state[preview_choice]
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

# ============================================================================
# TAB 2: DATA CLEANING
# ============================================================================
with tabs[1]:
    st.header("2️⃣ Data Cleaning & Quality Analysis")

    if st.session_state.listings is None:
        st.warning("⚠️ Upload data first!")
    else:
        # Cleaning options
        st.subheader("⚙️ Cleaning Configuration")

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

            # 1. Remove duplicates
            progress.progress(10)
            if opt_duplicates and 'id' in listings.columns:
                with st.expander("🔄 Duplicate Removal", expanded=True):
                    listings, dup_removed = remove_duplicates_with_log(listings, ['id'], logger)

                    stats = logger.get_stats().get("Duplicate Removal", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original", f"{stats.get('Original rows', 0):,}")
                    with col2:
                        st.metric("Removed", f"{stats.get('Rows removed', 0):,}")
                    with col3:
                        st.metric("Remaining", f"{stats.get('Final rows', 0):,}")

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

                        # Remove invalid
                        before = len(calendar)
                        calendar = calendar[calendar['price_cal'].notna()]
                        calendar = calendar[(calendar['price_cal'] > 0) & (calendar['price_cal'] < 5000)]
                        removed = before - len(calendar)

                        st.write(f"✅ Cleaned prices: removed {removed:,} invalid rows")

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

                # Fill remaining NaN
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64'] and df[col].isnull().sum() > 0:
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

            # Final feature list
            features = [
                'accommodates', 'bedrooms', 'beds', 'bathrooms',
                'latitude', 'longitude', 'neighbourhood_enc', 'dist_center',
                'room_type_enc', 'property_type_enc',
                'is_superhost', 'host_verified', 'has_profile_pic', 'host_listings_count',
                'host_response_rate', 'host_acceptance_rate',
                'instant_book', 'minimum_nights', 'amenities_count', 'desc_length', 'name_length',
                'has_luxury', 'premium_amenities',
                'review_scores_rating', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_value',
                'review_composite', 'review_min', 'review_std', 'number_of_reviews', 'reviews_per_month',
                'beds_per_bedroom', 'baths_per_bedroom', 'capacity_score', 'space_score',
                'cal_price_mean', 'cal_price_std', 'cal_price_min', 'cal_price_max', 'cal_price_median',
                'cal_price_q25', 'cal_price_q75', 'cal_avail_rate',
                'price_range', 'price_iqr', 'price_volatility', 'dynamic_pricing', 'high_demand',
                'review_count', 'avg_comment_len', 'avg_comment_words', 'max_comment_words',
                'total_positive', 'total_negative', 'avg_sentiment', 'sentiment_ratio',
                'days_since_review', 'has_recent_review',
                'neigh_centroid_lon', 'neigh_centroid_lat', 'neigh_area', 'neigh_perimeter',
                'neigh_group_enc', 'neigh_listing_count', 'neigh_density',
                'neigh_avg_rating', 'neigh_avg_accommodates', 'neigh_avg_bedrooms', 'neigh_avg_reviews',
                'dist_to_neigh_center'
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
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            results = {}
            trained = {}
            progress = st.progress(0)

            for i, name in enumerate(st.session_state.selected_models):
                st.write(f"Training {name}...")
                info = ALL_MODELS[name]
                model = type(info['model'])(**info['model'].get_params())

                X_tr = st.session_state.X_train_scaled if info['scale'] else st.session_state.X_train
                X_te = st.session_state.X_test_scaled if info['scale'] else st.session_state.X_test

                try:
                    model.fit(X_tr, st.session_state.y_train)
                    pred = model.predict(X_te)
                    r2 = r2_score(st.session_state.y_test, pred)
                    mae = mean_absolute_error(st.session_state.y_test, pred)
                    rmse = np.sqrt(mean_squared_error(st.session_state.y_test, pred))
                    results[name] = {'R2': r2, 'Accuracy': r2 * 100, 'MAE': mae, 'RMSE': rmse, 'predictions': pred}
                    trained[name] = {'model': model, 'needs_scale': info['scale']}
                except Exception as e:
                    results[name] = {'R2': 0, 'error': str(e)}

                progress.progress((i + 1) / len(st.session_state.selected_models))

            st.session_state.model_results = results
            st.session_state.trained_models = trained

            sorted_results = sorted([(k, v) for k, v in results.items() if 'error' not in v], key=lambda x: x[1]['R2'],
                                    reverse=True)
            df_results = pd.DataFrame(
                [{'Model': n, 'Accuracy': f"{r['Accuracy']:.2f}%", 'MAE': f"${r['MAE']:.2f}"} for n, r in
                 sorted_results])
            st.dataframe(df_results, use_container_width=True)

            if sorted_results[0][1]['R2'] >= 0.8:
                st.balloons()

with tabs[6]:
    st.header("7️⃣ Results")

    if not st.session_state.model_results:
        st.warning("⚠️ Train models first!")
    else:
        results = st.session_state.model_results
        sorted_results = sorted([(k, v) for k, v in results.items() if 'error' not in v], key=lambda x: x[1]['R2'],
                                reverse=True)

        df_comp = pd.DataFrame([{'Model': n, 'Accuracy (%)': round(r['Accuracy'], 2), 'MAE ($)': round(r['MAE'], 2),
                                 'RMSE ($)': round(r['RMSE'], 2)} for n, r in sorted_results])
        st.dataframe(df_comp, use_container_width=True)

        fig = px.bar(df_comp, x='Model', y='Accuracy (%)', color='Accuracy (%)', title="Model Comparison")
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

with tabs[7]:
    st.header("8️⃣ Price Prediction Tool")

    if not st.session_state.trained_models:
        st.warning("⚠️ Train models first!")
    else:
        # Get results and find best models
        results = st.session_state.model_results
        sorted_models = sorted([(k, v) for k, v in results.items() if 'error' not in v],
                               key=lambda x: x[1]['R2'], reverse=True)

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
                    st.metric("MAE", f"${metrics['MAE']:.2f}")
                    st.metric("RMSE", f"${metrics['RMSE']:.2f}")

            st.success(f"**Best Model: {best_model_name}** with R² = {sorted_models[0][1]['R2']:.2%}")

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

            # Get center from training data if available
            if st.session_state.X_train is not None and 'latitude' in st.session_state.X_train.columns:
                default_lat = st.session_state.X_train['latitude'].median()
                default_lon = st.session_state.X_train['longitude'].median()
            else:
                default_lat, default_lon = 46.0878, -64.7782

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

                        # Estimated base price for calendar features
                        estimated_price_base = 50 + (bedrooms * 25) + (accommodates * 10) + (bathrooms * 15)

                        # Create feature dictionary
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
                            'host_response_rate': 0.9,
                            'host_acceptance_rate': 0.9,
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
                            'cal_price_mean': estimated_price_base,
                            'cal_price_std': estimated_price_base * 0.15,
                            'cal_price_min': estimated_price_base * 0.7,
                            'cal_price_max': estimated_price_base * 1.5,
                            'cal_price_median': estimated_price_base,
                            'cal_price_q25': estimated_price_base * 0.85,
                            'cal_price_q75': estimated_price_base * 1.15,
                            'cal_avail_rate': 0.7,
                            'price_range': estimated_price_base * 0.8,
                            'price_iqr': estimated_price_base * 0.3,
                            'price_volatility': 0.15,
                            'dynamic_pricing': 0,
                            'high_demand': 0,
                            'review_count': number_of_reviews,
                            'avg_comment_len': 150,
                            'avg_comment_words': 30,
                            'max_comment_words': 100,
                            'total_positive': int(number_of_reviews * review_rating / 5 * 2),
                            'total_negative': int(number_of_reviews * (5 - review_rating) / 5 * 0.5),
                            'total_sentiment': int(number_of_reviews * (review_rating - 2.5)),
                            'avg_sentiment': (review_rating - 2.5) / 2.5,
                            'sentiment_ratio': review_rating / max(5 - review_rating, 0.5),
                            'days_since_review': 30,
                            'has_recent_review': 1 if number_of_reviews > 0 else 0,
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

                        # Create DataFrame with features
                        features = st.session_state.features
                        user_df = pd.DataFrame([{f: feature_values.get(f, 0) for f in features}])

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
                            predictions[model_name] = max(pred, 10)  # Ensure positive

                        # Calculate average
                        avg_pred = np.mean(list(predictions.values()))

                        # Display Results
                        st.markdown("---")
                        st.markdown("### 💰 Predicted Nightly Prices")

                        result_cols = st.columns(len(predictions) + 1)
                        for i, (model_name, pred) in enumerate(predictions.items()):
                            with result_cols[i]:
                                r2 = results[model_name]['R2']
                                badge = "🏆 " if model_name == best_model_name else ""
                                st.metric(label=f"{badge}{model_name}", value=f"${pred:,.2f}",
                                          help=f"R² = {r2:.2%}")

                        with result_cols[-1]:
                            st.metric(label="📊 Average", value=f"${avg_pred:,.2f}")

                        # Recommendation
                        st.info(f"""
                        **💡 Pricing Recommendation**

                        Based on your property characteristics, we recommend pricing your listing at approximately **${avg_pred:,.2f}** per night.

                        **Suggested Price Range:**
                        - 💰 **Budget-friendly** (faster bookings): ${avg_pred*0.85:,.2f} - ${avg_pred*0.95:,.2f}
                        - ⭐ **Competitive** (balanced): ${avg_pred*0.95:,.2f} - ${avg_pred*1.05:,.2f}
                        - 💎 **Premium** (peak season/events): ${avg_pred*1.1:,.2f} - ${avg_pred*1.25:,.2f}
                        """)

                        # Price factors breakdown
                        with st.expander("📈 What Factors Influenced This Price?"):
                            factors = []

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

            if st.button("🔍 Show Test Set Predictions", key="test_pred_btn"):
                best_info = st.session_state.trained_models[best_model_name]
                X_te = st.session_state.X_test_scaled if best_info['needs_scale'] else st.session_state.X_test
                pred = best_info['model'].predict(X_te)

                df_pred = pd.DataFrame({
                    'Actual': st.session_state.y_test.values,
                    'Predicted': pred.round(2),
                    'Error': (st.session_state.y_test.values - pred).round(2),
                    'Error %': ((st.session_state.y_test.values - pred) / st.session_state.y_test.values * 100).round(1)
                })

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(df_pred.head(50), use_container_width=True)
                with col2:
                    fig = px.scatter(df_pred.head(200), x='Actual', y='Predicted',
                                     title=f"Actual vs Predicted ({best_model_name})")
                    fig.add_trace(go.Scatter(x=[df_pred['Actual'].min(), df_pred['Actual'].max()],
                                             y=[df_pred['Actual'].min(), df_pred['Actual'].max()],
                                             mode='lines', name='Perfect Prediction',
                                             line=dict(dash='dash', color='red')))
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button("📥 Download Predictions", df_pred.to_csv(index=False), "predictions.csv")

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:gray'>🏠 Airbnb ML Studio | {len(ALL_MODELS)} Models | Advanced Cleaning & Logging</div>",
    unsafe_allow_html=True)