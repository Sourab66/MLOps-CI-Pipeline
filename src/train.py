from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train():
    X, y = load_iris(return_X_y=True)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    return model


if __name__ == "__main__":
    train()
