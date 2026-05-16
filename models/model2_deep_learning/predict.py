#!/usr/bin/env python3
"""
Model 2: Deep Learning — Prediction Script
============================================
Loads the trained DNN and generates binary severity predictions on test data.

Usage: python predict.py
Output: test_data/model2_results.csv
"""
import sys
import pandas as pd
import platform
from pathlib import Path
import tensorflow as tf
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import clean_data, engineer_features

# Paths
MODEL_PATH     = Path("models/model2_deep_learning/saved_model/")
TEST_DATA_DIR  = Path("test_data/")
OUTPUT_FILE    = TEST_DATA_DIR / "model2_results.csv"
TEST_DATA_FILE = TEST_DATA_DIR / "city_traffic_accidents.csv"

# NOTE: This list intentionally matches the 29-column order used at training time.
# wind_dir_deg, weather_cond_num, and accident_dir appear twice — the scaler was
# fit on exactly this 29-column layout, so the order must be preserved here.
features = [
    'Distance(mi)', 'Timezone',
    'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
    'wind_dir_deg', 'weather_cond_num', 'accident_dir',
    'hour', 'day_of_week', 'month', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
    'duration_min', 'wind_dir_deg', 'weather_cond_num', 'weather_data_available',
    'is_freezing', 'low_visibility', 'accident_dir', 'lat_bin',
    'n_road_features', 'has_traffic_control',
]

def is_apple_silicon():
    return (
        platform.system() == "Darwin" and
        platform.machine() == "arm64"
    )

def disable_gpus():
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Disable all GPUs by setting the visible device list to empty
            tf.config.set_visible_devices([], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def load_model():
    model  = tf.keras.models.load_model(MODEL_PATH / "model.keras")
    scaler = joblib.load(MODEL_PATH / "scaler.joblib")
    return model, scaler

def preprocess(df):
    """Apply the same cleaning and feature engineering used during training."""
    ids = df['ID'].copy() if 'ID' in df.columns else pd.Series(range(len(df)))

    df = clean_data(df)
    df = engineer_features(df)

    # Select feature columns (29 cols, including intentional duplicates for scaler alignment)
    X = df[features].copy().astype(np.float32).fillna(0.0)
    return ids, X

def predict(model, X):
    """Generate predictions on test data.

    Should return a DataFrame with columns: id, prediction, probability, confidence
    """
    predictions = model.predict(X)
    return predictions

def main():
    if is_apple_silicon():
        print("Running on Apple Silicon — disabling GPU to avoid compatibility issues.")
        disable_gpus()

    # Load model
    model, scaler = load_model()

    # Load test data
    print(f"Loading test data from {TEST_DATA_FILE}...")
    test_df = pd.read_csv(TEST_DATA_FILE)
    print(f"  {len(test_df):,} rows loaded")

    # Preprocess raw data
    print("Preprocessing...")
    ids, X = preprocess(test_df)

    # Generate predictions
    scaled_X = scaler.transform(X)
    predictions = predict(model, scaled_X)

    # Binary classifier (sigmoid): model outputs positive-class probability per row.
    positive_probs = predictions.ravel()
    predicted_labels = (positive_probs >= 0.5).astype(int)
    confidence_scores = np.where(predicted_labels == 1, positive_probs, 1 - positive_probs)

    # Save results — MUST match output template exactly
    results = pd.DataFrame({
        "id":          ids.values,
        "prediction":  predicted_labels,
        "probability": positive_probs.round(4),
        "confidence":  confidence_scores.round(4),
    })
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE} ({len(results):,} rows)")


if __name__ == "__main__":
    main()
