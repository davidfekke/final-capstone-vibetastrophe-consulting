#!/usr/bin/env python3
"""
Model 4: NLP Classification — Prediction Script
=================================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model4_results.csv
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Paths
MODEL_PATH     = Path("models/model4_nlp_classification/saved_model/")
TEST_DATA_DIR  = Path("test_data/")
OUTPUT_FILE    = TEST_DATA_DIR / "model4_results.csv"
TEST_DATA_FILE = TEST_DATA_DIR / "urbanpulse_311_complaints.csv"

LABELS = {
    0: "Blocked Driveway",
    1: "HEAT/HOT WATER",
    2: "Illegal Parking",
    3: "Noise - Residential",
    4: "Other",
    5: "Snow or Ice",
}


def load_model():
    """Load the trained GRU model and vectorizer from saved_model/."""
    import tensorflow as tf
    model         = tf.keras.models.load_model(MODEL_PATH / "gru_model.keras")
    vectorizer    = joblib.load(MODEL_PATH / "vectorizer.joblib")
    label_encoder = joblib.load(MODEL_PATH / "label_encoder.joblib")
    return model, vectorizer, label_encoder


def preprocess_text(texts):
    texts = texts.fillna("").astype(str).str.lower()
    texts = texts.str.replace(r"[^\w\s]", "", regex=True)
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()
    return texts


def predict(model, vectorizer, texts, batch_size=512):
    """Generate predictions on text data.

    Should return a tuple of (predicted_classes, confidence_scores).
    """
    import tensorflow as tf
    all_probs  = []
    texts_list = texts.tolist()

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i : i + batch_size]
        vecs  = vectorizer(tf.constant(batch, dtype=tf.string))
        probs = model.predict(vecs, verbose=0)
        all_probs.append(probs)
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,} / {len(texts_list):,} processed")

    all_probs     = np.vstack(all_probs)
    predicted_ids = np.argmax(all_probs, axis=1)
    confidence    = all_probs.max(axis=1).round(4)
    labels        = [LABELS.get(int(idx), "Other") for idx in predicted_ids]
    return labels, confidence


def main():
    # Load model
    model, vectorizer, label_encoder = load_model()

    # Load test data
    test_df = pd.read_csv(TEST_DATA_FILE)

    # Preprocess text
    texts = preprocess_text(test_df["resolution_description"])

    # Generate predictions
    predicted_classes, confidence_scores = predict(model, vectorizer, texts)

    # Save results — MUST match output template exactly
    results = pd.DataFrame({
        "id":              test_df["unique_key"],
        "predicted_class": predicted_classes,
        "confidence":      confidence_scores,
    })
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
