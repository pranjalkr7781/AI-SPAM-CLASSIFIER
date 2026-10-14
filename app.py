
import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Spam Classifier",
    page_icon="📩",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

st.title("📩 AI Spam Classifier")
st.write(
    "Enter an SMS or email message and check whether "
    "it is likely to be spam or legitimate."
)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(
        "Could not load the trained model. "
        "Check that both model files exist and "
        "their dependencies are compatible."
    )
    st.exception(e)
    st.stop()

message = st.text_area(
    "Enter your message",
    placeholder="Congratulations! You have won a free prize...",
    height=150
)

if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        vectorized_message = vectorizer.transform([message])
        prediction = model.predict(vectorized_message)[0]

        st.subheader("Prediction")

        if str(prediction).lower() == "spam":
            st.error("🚨 This message is classified as SPAM.")
        else:
            st.success("✅ This message is classified as NOT SPAM.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                vectorized_message
            )[0]

            confidence = max(probabilities) * 100

            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

st.divider()
st.caption(
    "Built with Python, TF-IDF and Multinomial Naive Bayes."
)
