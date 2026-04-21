"""
Phase 4: Train the AI Model
Reads data/signs.csv and trains a Random Forest classifier.
Saves the model to models/sign_model.pkl.
"""

import os
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'signs.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sign_model.pkl')


def load_data():
    labels = []
    features = []

    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])

    return np.array(features), np.array(labels)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: No training data found at {DATA_PATH}")
        print("Run src/collect_data.py first to record signs.")
        return

    print("Loading data...")
    X, y = load_data()
    print(f"  {len(X)} samples loaded, {len(set(y))} unique signs: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()
