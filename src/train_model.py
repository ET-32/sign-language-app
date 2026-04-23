"""
Train the sign language model.
Reads data/signs.csv and trains a classifier.
Saves to models/sign_model.pkl.
"""

import os
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'signs.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sign_model.pkl')


def load_data():
    labels, features = [], []
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])
    return np.array(features), np.array(labels)


def main():
    if not os.path.exists(DATA_PATH):
        print(f'ERROR: No training data at {DATA_PATH}')
        print('Run: py -m src.collect_data')
        return

    print('Loading data...')
    X, y = load_data()
    classes = sorted(set(y))
    print(f'  {len(X)} samples, {len(classes)} signs: {classes}')

    counts = {c: int((y == c).sum()) for c in classes}
    min_count = min(counts.values())
    if min_count < 20:
        low = [c for c, n in counts.items() if n < 20]
        print(f'WARNING: {low} have fewer than 20 samples — collect more for better accuracy.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SVM with RBF kernel — better than Random Forest for hand landmark data
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(kernel='rbf', C=10, gamma='scale', probability=True)),
    ])

    print('Training SVM model...')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f'\nTest accuracy: {acc * 100:.1f}%')
    print('\nPer-sign results:')
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
