"""Inference module for email spam detection."""

import os
import pickle
from typing import Dict, Any, Tuple
from src.preprocess import preprocess_text

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODEL_PATH = os.path.join(_MODELS_DIR, "model.pkl")
_VECTORIZER_PATH = os.path.join(_MODELS_DIR, "vectorizer.pkl")

# Cached model & vectorizer
_CACHED_MODEL = None
_CACHED_VECTORIZER = None


def load_artifacts() -> Tuple[Any, Any]:
    """Load model and vectorizer from pickle files.

    Returns:
        Tuple[Any, Any]: (model, vectorizer)

    Raises:
        FileNotFoundError: If model.pkl or vectorizer.pkl does not exist.
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER

    if _CACHED_MODEL is not None and _CACHED_VECTORIZER is not None:
        return _CACHED_MODEL, _CACHED_VECTORIZER

    if not os.path.exists(_MODEL_PATH) or not os.path.exists(_VECTORIZER_PATH):
        raise FileNotFoundError(
            "Trained model or vectorizer artifact not found. "
            "Please train the model first by running: python train.py"
        )

    with open(_MODEL_PATH, "rb") as f:
        _CACHED_MODEL = pickle.load(f)

    with open(_VECTORIZER_PATH, "rb") as f:
        _CACHED_VECTORIZER = pickle.load(f)

    return _CACHED_MODEL, _CACHED_VECTORIZER


def predict_email(text: str) -> Dict[str, Any]:
    """Predict whether an email is spam or normal (ham).

    Applies text preprocessing, vectorization, and model inference.

    Args:
        text (str): Raw email content.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "label": "Spam" or "Normal"
            - "is_spam": bool (True if spam)
            - "spam_probability": float (0.0 to 1.0)
            - "ham_probability": float (0.0 to 1.0)
            - "confidence": float (percentage confidence)
            - "processed_text": str
    """
    if not text or not text.strip():
        return {
            "label": "Unknown",
            "is_spam": False,
            "spam_probability": 0.0,
            "ham_probability": 0.0,
            "confidence": 0.0,
            "processed_text": "",
        }

    model, vectorizer = load_artifacts()

    # Preprocess with identical pipeline
    processed = preprocess_text(text)
    if not processed:
        # If all tokens were stripped (e.g. only numbers or punctuation)
        processed = ""

    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    # Class 0: Ham/Normal, Class 1: Spam
    classes = list(model.classes_)
    spam_idx = classes.index(1) if 1 in classes else 1
    ham_idx = classes.index(0) if 0 in classes else 0

    spam_prob = float(probabilities[spam_idx])
    ham_prob = float(probabilities[ham_idx])

    is_spam = bool(prediction == 1)
    label = "Spam" if is_spam else "Normal"
    confidence = spam_prob if is_spam else ham_prob

    return {
        "label": label,
        "is_spam": is_spam,
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4),
        "confidence": round(confidence * 100, 2),
        "processed_text": processed,
    }
