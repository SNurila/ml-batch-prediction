"""
train_model.py
Trains a simple Iris classifier and saves it to models/iris_model.pkl.
Run once before starting the pipeline.
"""

import os
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "iris_model.pkl")


def train_and_save():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[train_model] Test accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "target_names": list(iris.target_names)}, f)

    print(f"[train_model] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()