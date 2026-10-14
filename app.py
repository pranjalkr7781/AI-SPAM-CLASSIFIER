
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"
DATA_PATH = BASE_DIR / "data" / "sms_data.csv"

st.set_page_config(
    page_title="AI Spam Classifier",
    page_icon="🛡️",
    layout="wide",
)

# -------------------- CUSTOM DESIGN --------------------

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
[data-testid="stSidebar"] {
    background: #101a2b;
    border-right: 1px solid #263449;
}
div[data-testid="stMetric"] {
    background: #152238;
    border: 1px solid #293951;
    padding: 16px;
    border-radius: 12px;
}
div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}
.panel {
    background: #111c2e;
    border: 1px solid #293951;
    padding: 20px;
    border-radius: 13px;
    margin-bottom: 14px;
}
.safe {
    background: #10352d;
    border: 1px solid #16845e;
    border-radius: 12px;
    padding: 18px;
}
.danger {
    background: #3a1b26;
    border: 1px solid #a33b52;
    border-radius: 12px;
    padding: 18px;
}
.muted {
    color: #9caec5;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


try:
    model, vectorizer = load_artifacts()
    model_ready = True
    model_error = None
except Exception as exc:
    model_ready = False
    model_error = str(exc)
    model = None
    vectorizer = None

if "history" not in st.session_state:
    st.session_state.history = []

if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

if "spam_count" not in st.session_state:
    st.session_state.spam_count = 0


# -------------------- PREDICTION HELPERS --------------------

def is_spam_label(label):
    """Support common spam/ham and spam/not-spam labels."""
    value = str(label).strip().lower()
    spam_labels = {"spam", "1", "true", "junk", "phishing"}
    return value in spam_labels


def predict_message(message):
    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    spam = is_spam_label(prediction)

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities)) * 100

    return spam, confidence


def record_scan(message, spam, confidence):
    st.session_state.total_scans += 1
    if spam:
        st.session_state.spam_count += 1

    st.session_state.history.insert(0, {
        "Time": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        "Message": message,
        "Prediction": "Spam" if spam else "Not Spam",
        "Confidence": (
            f"{confidence:.1f}%"
            if confidence is not None else "N/A"
        ),
    })

    # Keep recent history manageable.
    st.session_state.history = st.session_state.history[:50]


def show_result(message, spam, confidence):
    css_class = "danger" if spam else "safe"
    icon = "🚨" if spam else "✅"
    title = "SPAM DETECTED" if spam else "NOT SPAM"
    description = (
        "This message was classified as likely spam."
        if spam else
        "This message was classified as likely legitimate."
    )

    st.markdown(
        f'<div class="{css_class}">'
        f'<h3>{icon} {title}</h3>'
        f'<p>{description}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if confidence is not None:
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee of correctness."
        )

    with st.expander("View analyzed message"):
        st.write(message)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("# 🛡️ AI Spam Classifier")
    st.caption("SMS & Email Protection")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Message Scanner",
            "Batch Analysis",
            "Dataset Info",
            "About",
        ],
    )

    st.divider()
    st.markdown("**System Status**")
    if model_ready:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")

    st.caption("Python · Scikit-learn · Streamlit")


# -------------------- HEADER --------------------

st.title("🛡️ AI Spam Classifier")
st.caption(
    "Check SMS and email messages using machine learning."
)

if not model_ready:
    st.error(
        "The trained model could not be loaded. Check that "
        "`model/spam_model.pkl` and `model/vectorizer.pkl` "
        "exist and that your installed dependencies match "
        "the versions used during training."
    )
    with st.expander("Technical error"):
        st.code(model_error)
    st.stop()


# -------------------- MESSAGE SCANNER --------------------

if page == "Message Scanner":
    left, right = st.columns([1.8, 1])

    with left:
        st.markdown("### Analyze a message")

        message = st.text_area(
            "SMS or email content",
            placeholder=(
                "Example: Congratulations! You have won a prize..."
            ),
            height=170,
            max_chars=10000,
            key="single_message",
        )

        st.caption(f"{len(message)} characters")

        st.markdown("**Try an example**")
        examples = {
            "Prize message": (
                "Congratulations! You have won a free prize. "
                "Click here now!"
            ),
            "Normal conversation": (
                "Hi, are we still meeting at the library at 4 PM?"
            ),
            "Account warning": (
                "Your account needs attention. Verify your details."
            ),
        }

        example_choice = st.selectbox(
            "Load sample",
            ["Choose an example"] + list(examples.keys()),
        )

       
if st.button("Load Example"):
    if example_choice in examples:
        st.session_state["pending_example"] = examples[
            example_choice
        ]
        st.rerun()

if "pending_example" in st.session_state:
    st.session_state["single_message"] = (
        st.session_state.pop("pending_example")
    )

        if st.button("🔍 Analyze Message", use_container_width=True):
            if not message.strip():
                st.warning("Please enter a message first.")
            else:
                try:
                    spam, confidence = predict_message(message)
                    record_scan(message, spam, confidence)
                    st.session_state.last_result = {
                        "message": message,
                        "spam": spam,
                        "confidence": confidence,
                    }
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        if "last_result" in st.session_state:
            st.markdown("### Analysis Result")
            result = st.session_state.last_result
            show_result(
                result["message"],
                result["spam"],
                result["confidence"],
            )

    with right:
        st.markdown("### Statistics")

        total = st.session_state.total_scans
        spam_total = st.session_state.spam_count
        safe_total = total - spam_total
        spam_rate = spam_total / total * 100 if total else 0

        a, b = st.columns(2)
        a.metric("Total scans", total)
        b.metric("Spam detected", spam_total)

        c, d = st.columns(2)
        c.metric("Not spam", safe_total)
        d.metric("Spam rate", f"{spam_rate:.1f}%")

        st.markdown("### Recent Scans")
        if st.session_state.history:
            history_df = pd.DataFrame(
                st.session_state.history
            )
            st.dataframe(
                history_df[
                    ["Time", "Message", "Prediction"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Your scan history will appear here.")

        st.caption(
            "Statistics and history are for this active session."
        )


# -------------------- BATCH ANALYSIS --------------------

elif page == "Batch Analysis":
    st.markdown("### Analyze multiple messages")
    st.write(
        "Upload a CSV file with a column named `text`. "
        "Each row will be classified using your trained model."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            if "text" not in batch_df.columns:
                st.error(
                    "The CSV must contain a column named `text`."
                )
            elif batch_df.empty:
                st.warning("The uploaded CSV has no rows.")
            elif batch_df["text"].isna().any():
                st.error(
                    "Some messages are empty. Clean the CSV "
                    "before running batch analysis."
                )
            elif len(batch_df) > 10000:
                st.error(
                    "Please upload at most 10,000 rows at a time."
                )
            else:
                st.write("Preview")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button("Analyze CSV", use_container_width=True):
                    texts = batch_df["text"].astype(str).tolist()
                    features = vectorizer.transform(texts)
                    predictions = model.predict(features)

                    results = batch_df.copy()
                    results["Prediction"] = [
                        "Spam" if is_spam_label(p) else "Not Spam"
                        for p in predictions
                    ]

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)
                        results["Confidence (%)"] = (
                            probabilities.max(axis=1) * 100
                        ).round(2)

                    st.success(
                        f"Successfully analyzed {len(results)} messages."
                    )

                    x, y, z = st.columns(3)
                    x.metric("Messages", len(results))
                    y.metric(
                        "Spam",
                        int((results["Prediction"] == "Spam").sum()),
                    )
                    z.metric(
                        "Not Spam",
                        int(
                            (results["Prediction"] == "Not Spam").sum()
                        ),
                    )

                    st.dataframe(results, use_container_width=True)

                    st.download_button(
                        "⬇️ Download Results",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="spam_analysis_results.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")


# -------------------- DATASET INFORMATION --------------------

elif page == "Dataset Info":
    st.markdown("### Dataset Overview")

    try:
        df = load_dataset()

        if df.empty:
            st.warning(
                "Dataset not found at `data/sms_data.csv`."
            )
        elif not {"label", "text"}.issubset(df.columns):
            st.warning(
                "Expected dataset columns: `label` and `text`."
            )
        else:
            total_rows = len(df)
            spam_rows = df["label"].astype(str).str.lower().isin(
                ["spam", "1", "true", "junk"]
            ).sum()
            ham_rows = total_rows - spam_rows

            a, b, c = st.columns(3)
            a.metric("Total messages", total_rows)
            b.metric("Spam messages", int(spam_rows))
            c.metric("Other messages", int(ham_rows))

            st.markdown("### Label Distribution")
            counts = df["label"].astype(str).value_counts()
            st.bar_chart(counts)

            st.markdown("### Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")

    st.markdown("### Model Information")
    st.markdown("""
    - **Text vectorization:** TF-IDF
    - **Classification algorithm:** Multinomial Naive Bayes
    - **Model file:** `model/spam_model.pkl`
    - **Vectorizer file:** `model/vectorizer.pkl`
    """)


# -------------------- ABOUT --------------------

elif page == "About":
    st.markdown("### About this project")

    st.markdown("""
    This project classifies SMS and email text as spam or not spam.

    **How it works**

    1. The user provides a message.
    2. TF-IDF converts the text into numerical features.
    3. Multinomial Naive Bayes predicts the message category.
    4. The application displays the prediction and, when
       supported, the model's estimated confidence.

    **Technology stack**
    - Python
    - Pandas
    - Scikit-learn
    - Joblib
    - Streamlit

    **Limitations**

    This application predicts message categories. It does not
    automatically read, delete, or block messages in your inbox.
    Results depend on the training data and model quality.
    """)

st.divider()
st.caption(
    "AI Spam Classifier · Built with Python, Scikit-learn and Streamlit"
)
