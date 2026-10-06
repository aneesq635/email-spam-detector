"""Streamlit web interface for Email Spam Detection."""

import os
import pickle
import streamlit as st
from src.predict import predict_email

# Set Streamlit page configuration
st.set_page_config(
    page_title="Email Spam Classifier | NLP Assignment",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.pkl")
CM_IMAGE_PATH = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")


def load_metrics():
    """Load saved evaluation metrics if available."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


# Sidebar
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shield-checked.png",
        width=70,
    )
    st.title("Model & NLP Info")
    st.markdown(
        """
        **Assignment Project**  
        Natural Language Processing (NLP)  
        **Model:** Multinomial Naive Bayes  
        **Feature Extractor:** Bag of Words (`CountVectorizer`, max 5000)
        """
    )
    st.markdown("---")

    metrics = load_metrics()
    if metrics:
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        with col2:
            st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
            st.metric("F1-Score", f"{metrics['f1'] * 100:.1f}%")

        st.caption(f"Trained on {metrics.get('train_samples', 'N/A')} samples, tested on {metrics.get('test_samples', 'N/A')}.")
        st.caption(f"Vocabulary size: {metrics.get('vocab_size', 'N/A')} tokens.")
    else:
        st.info("Performance metrics will appear here once `python train.py` is executed.")

    st.markdown("---")
    st.subheader("💡 Sample Test Emails")
    sample_choice = st.selectbox(
        "Load a sample email into editor:",
        [
            "-- Select sample --",
            "Urgent: Claim Your $1,000,000 Lottery Prize Now! (Spam)",
            "Special Discount Offer: 80% off watches today only! (Spam)",
            "Project update: Weekly sprint meeting on Thursday (Normal)",
            "Team dinner reservation details for Friday night (Normal)",
        ],
    )


# Main layout
st.title("🛡️ Email Spam Detection System")
st.markdown(
    """
    Classify emails in real-time as **Normal (Ham)** or **Spam** using an NLP pipeline powered by 
    NLTK tokenization, Porter stemming, Bag-of-Words feature extraction, and Multinomial Naive Bayes.
    """
)

# Check if model artifacts exist
artifacts_present = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)

if not artifacts_present:
    st.error(
        "⚠️ **Trained model artifacts not found!**\n\n"
        "Please place your dataset CSV in the `data/` folder and execute the training script:\n\n"
        "```bash\npython train.py\n```\n"
        "Once trained, refresh this page."
    )
else:
    # Populate sample text if selected
    default_text = ""
    if sample_choice == "Urgent: Claim Your $1,000,000 Lottery Prize Now! (Spam)":
        default_text = (
            "Congratulations! You have been selected as the winner of the global lottery $1,000,000 prize. "
            "Reply immediately with your bank account details to claim your cash award now!"
        )
    elif sample_choice == "Special Discount Offer: 80% off watches today only! (Spam)":
        default_text = (
            "Exclusive offer! Buy replica luxury watches for 80% off. Limited stock remaining. "
            "Click here: http://discount-watches-fake.com to grab yours before deal expires!"
        )
    elif sample_choice == "Project update: Weekly sprint meeting on Thursday (Normal)":
        default_text = (
            "Hi team, please find attached the agenda for our weekly sprint review on Thursday at 2 PM. "
            "Let me know if you have any questions or items to add before tomorrow."
        )
    elif sample_choice == "Team dinner reservation details for Friday night (Normal)":
        default_text = (
            "Hey all, I've booked a table for 8 at the Italian bistro for this Friday at 7:30 PM. "
            "Looking forward to celebrating our project milestone with everyone."
        )

    # Input area
    user_input = st.text_area(
        "Paste or type the email message text below:",
        value=default_text,
        height=180,
        placeholder="Type or paste the email content here...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        check_clicked = st.button("🔍 Check Email", type="primary", use_container_width=True)

    if check_clicked:
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter email text before clicking 'Check Email'.")
        else:
            try:
                result = predict_email(user_input)

                st.markdown("### Prediction Result")

                if result["is_spam"]:
                    st.error(
                        f"🚨 **SPAM DETECTED**\n\n"
                        f"Confidence: **{result['confidence']:.2f}%** | "
                        f"Spam Probability: `{result['spam_probability']}`"
                    )
                else:
                    st.success(
                        f"✅ **NORMAL (Not Spam)**\n\n"
                        f"Confidence: **{result['confidence']:.2f}%** | "
                        f"Ham Probability: `{result['ham_probability']}`"
                    )

                with st.expander("🔍 View Preprocessing & Pipeline Details"):
                    st.markdown("**Processed & Stemmed Tokens Passed to Vectorizer:**")
                    st.code(result["processed_text"] or "(empty after stopword removal)", language="text")
                    st.markdown(
                        f"- **Raw Spam Probability:** `{result['spam_probability']}`\n"
                        f"- **Raw Normal (Ham) Probability:** `{result['ham_probability']}`\n"
                    )

            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Bottom section: Confusion Matrix if available
if os.path.exists(CM_IMAGE_PATH):
    with st.expander("📈 View Model Confusion Matrix (from outputs/)", expanded=False):
        st.image(CM_IMAGE_PATH, caption="Confusion Matrix from last training run", width=500)
