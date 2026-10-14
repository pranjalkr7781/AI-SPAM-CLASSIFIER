from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"
DATA_PATH = BASE_DIR / "data" / "sms_data.csv"

st.set_page_config(
    page_title="AI Spam Classifier",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111c2e;
    border-right: 1px solid #293951;
}
.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}
div[data-testid="stMetric"] {
    background: #152238;
    border: 1px solid #293951;
    border-radius: 12px;
    padding: 14px;
}
div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 40px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}
.safe-result {
    background: #10352d;
    border: 1px solid #16845e;
    border-radius: 12px;
    padding: 18px;
}
.spam-result {
    background: #3a1b26;
    border: 1px solid #a33b52;
    border-radius: 12px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def is_spam_label(label):
    value = str(label).strip().lower()
    return value in {"spam", "junk", "phishing", "true"}


def predict_message(message):
    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    spam = is_spam_label(prediction)

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        index = classes.index(prediction)
        confidence = float(probabilities[index]) * 100

    return spam, confidence


def record_scan(message, spam, confidence):
    st.session_state.total_scans += 1
    if spam:
        st.session_state.spam_count += 1

    st.session_state.history.insert(0, {
        "Time": datetime.now().strftime("%d %b %H:%M:%S"),
        "Message": message,
        "Prediction": "Spam" if spam else "Not Spam",
        "Confidence": (
            f"{confidence:.1f}%" if confidence is not None else "N/A"
        ),
    })
    st.session_state.history = st.session_state.history[:50]


def analyze_and_record(message):
    spam, confidence = predict_message(message)
    record_scan(message, spam, confidence)
    st.session_state.last_result = {
        "message": message,
        "spam": spam,
        "confidence": confidence,
    }


def show_result(result):
    if result["spam"]:
        st.markdown("""
        <div class="spam-result">
            <h3>🚨 Spam Detected</h3>
            <p>This message was classified as likely spam.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="safe-result">
            <h3>✅ Not Spam</h3>
            <p>This message was classified as likely legitimate.</p>
        </div>
        """, unsafe_allow_html=True)

    if result["confidence"] is not None:
        st.metric("Model Confidence", f'{result["confidence"]:.1f}%')
        st.caption("Estimated probability from the trained model.")

    with st.expander("View analyzed message"):
        st.write(result["message"])


if "history" not in st.session_state:
    st.session_state.history = []

if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

if "spam_count" not in st.session_state:
    st.session_state.spam_count = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "single_message" not in st.session_state:
    st.session_state.single_message = ""

try:
    model, vectorizer = load_model()
    model_ready = True
    model_error = None
except Exception as error:
    model_ready = False
    model_error = str(error)
    model = None
    vectorizer = None

with st.sidebar:
    st.markdown("# 🛡️ AI Spam Classifier")
    st.caption("SMS & Email Protection")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Message Scanner", "Batch Analysis", "Dataset Info", "About"],
    )

    st.divider()
    st.markdown("### System Status")

    if model_ready:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")

    st.caption("Python · Scikit-learn · Streamlit")

st.title("🛡️ AI Spam Classifier")
st.caption("Analyze SMS and email messages using machine learning.")

if not model_ready:
    st.error("The trained model could not be loaded.")
    st.code(model_error)
    st.stop()


if page == "Message Scanner":
    left, right = st.columns([1.7, 1])

    with left:
        st.subheader("Message Scanner")

        examples = {
            "Prize Message": (
                "Congratulations! You have won a free prize. "
                "Click here now!"
            ),
            "Normal Message": (
                "Hi, are we still meeting at the library at 4 PM?"
            ),
            "Account Warning": (
                "Your account needs attention. Verify your details."
            ),
        }

        example_clicked = None
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎁 Prize Message", use_container_width=True):
                example_clicked = examples["Prize Message"]

        with col2:
            if st.button("💬 Normal Message", use_container_width=True):
                example_clicked = examples["Normal Message"]

        with col3:
            if st.button("⚠️ Account Warning", use_container_width=True):
                example_clicked = examples["Account Warning"]

        if example_clicked is not None:
            st.session_state["single_message"] = example_clicked

        message = st.text_area(
            "SMS or email content",
            placeholder="Type or paste your message here...",
            height=160,
            max_chars=10000,
            key="single_message",
        )

        st.caption(f"{len(message)} characters")

        analyze_clicked = st.button(
            "🔍 Analyze Message",
            use_container_width=True,
        )

        if example_clicked is not None or analyze_clicked:
            text_to_analyze = (
                example_clicked if example_clicked is not None else message
            )

            if not text_to_analyze.strip():
                st.warning("Please enter a message first.")
            else:
                try:
                    analyze_and_record(text_to_analyze)
                except Exception as error:
                    st.error(f"Prediction failed: {error}")

        if st.session_state.last_result is not None:
            st.subheader("Analysis Result")
            show_result(st.session_state.last_result)

    with right:
        st.subheader("Statistics")

        total = st.session_state.total_scans
        spam_total = st.session_state.spam_count
        safe_total = total - spam_total
        spam_rate = spam_total / total * 100 if total else 0

        col1, col2 = st.columns(2)
        col1.metric("Total Scans", total)
        col2.metric("Spam Detected", spam_total)

        col3, col4 = st.columns(2)
        col3.metric("Not Spam", safe_total)
        col4.metric("Spam Rate", f"{spam_rate:.1f}%")

        st.subheader("Recent Scans")

        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Your scan history will appear here.")

        st.caption("Statistics are maintained for the current session.")


elif page == "Batch Analysis":
    st.subheader("Batch Analysis")
    st.write("Upload a CSV containing a column named `text`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            if "text" not in batch_df.columns:
                st.error("The CSV must contain a column named 'text'.")
            elif batch_df.empty:
                st.warning("The CSV contains no messages.")
            elif batch_df["text"].isna().any():
                st.error("Remove empty messages from the CSV first.")
            elif len(batch_df) > 10000:
                st.error("Upload a maximum of 10,000 messages at a time.")
            else:
                st.write("Preview")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button("Analyze CSV", use_container_width=True):
                    texts = batch_df["text"].astype(str).tolist()
                    features = vectorizer.transform(texts)
                    predictions = model.predict(features)

                    results = batch_df.copy()
                    results["Prediction"] = [
                        "Spam" if is_spam_label(item) else "Not Spam"
                        for item in predictions
                    ]

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)
                        classes = list(model.classes_)
                        indices = [
                            classes.index(item) for item in predictions
                        ]
                        results["Confidence (%)"] = [
                            round(float(probabilities[i][j]) * 100, 2)
                            for i, j in enumerate(indices)
                        ]

                    spam_count = int(
                        (results["Prediction"] == "Spam").sum()
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Messages", len(results))
                    col2.metric("Spam", spam_count)
                    col3.metric("Not Spam", len(results) - spam_count)

                    st.dataframe(results, use_container_width=True)

                    st.download_button(
                        "Download Results",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="spam_analysis_results.csv",
                        mime="text/csv",
                    )

        except Exception as error:
            st.error(f"Could not process the CSV: {error}")


elif page == "Dataset Info":
    st.subheader("Dataset Information")

    try:
        dataset = load_dataset()

        if dataset.empty:
            st.warning("Dataset not found at data/sms_data.csv.")
        elif not {"label", "text"}.issubset(dataset.columns):
            st.warning("Expected columns: label and text.")
        else:
            labels = dataset["label"].astype(str).str.strip().str.lower()
            spam_mask = labels.isin(
                ["spam", "1", "true", "junk", "phishing"]
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Messages", len(dataset))
            col2.metric("Spam Messages", int(spam_mask.sum()))
            col3.metric("Other Messages", int((~spam_mask).sum()))

            st.subheader("Label Distribution")
            st.bar_chart(dataset["label"].astype(str).value_counts())

            st.subheader("Dataset Preview")
            st.dataframe(dataset.head(20), use_container_width=True)

    except Exception as error:
        st.error(f"Could not read the dataset: {error}")

    st.subheader("Model Information")
    st.write("Algorithm: Multinomial Naive Bayes")
    st.write("Text vectorization: TF-IDF")
    st.write("Classifier: model/spam_model.pkl")
    st.write("Vectorizer: model/vectorizer.pkl")


elif page == "About":
    st.subheader("About This Project")

    st.write(
        "This application classifies SMS and email messages as spam "
        "or not spam using a trained machine learning model."
    )

    st.markdown("### How It Works")
    st.markdown("""
    1. The user enters a message.
    2. TF-IDF converts the text into numerical features.
    3. Multinomial Naive Bayes predicts the message category.
    4. The application displays the prediction and confidence.
    """)

    st.markdown("### Technologies")
    st.write("Python, Pandas, Scikit-learn, Joblib and Streamlit.")

    st.markdown("### Limitations")
    st.write(
        "Predictions depend on the training data. The application "
        "does not automatically read or block messages in an inbox."
    )

st.divider()
st.caption("AI Spam Classifier · Python · Scikit-learn · Streamlit")
