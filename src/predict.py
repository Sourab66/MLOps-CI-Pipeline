import joblib


def predict(sample):
    model = joblib.load("models/model.pkl")
    return model.predict([sample])


if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]
    print(predict(sample))
