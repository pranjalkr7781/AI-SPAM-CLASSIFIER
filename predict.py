"""Classify SMS/email text as spam or ham using the trained model.

Usage:
    python predict.py "Your message text here"
    python predict.py            # runs interactive mode
"""
import sys
import joblib

MODEL_PATH = "model/spam_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def load():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def classify(text, model, vectorizer):
    vec = vectorizer.transform([text])
    label = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    confidence = prob[classes.index(label)]
    return label, confidence


def main():
    model, vectorizer = load()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, confidence = classify(text, model, vectorizer)
        print(f"Message: {text}")
        print(f"Prediction: {label.upper()}  (confidence: {confidence:.2%})")
        return

    print("Spam Classifier - type a message and press Enter (or 'quit' to exit)")
    while True:
        text = input("\n> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        label, confidence = classify(text, model, vectorizer)
        print(f"Prediction: {label.upper()}  (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
