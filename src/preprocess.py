"""Text preprocessing pipeline for email spam classification.

Performs:
- Lowercasing
- HTML & URL removal
- Number & punctuation removal
- NLTK Tokenization
- Stopwords filtering
- Stemming (PorterStemmer)
"""

import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def ensure_nltk_resources() -> None:
    """Download required NLTK resources automatically if not already present."""
    resources = ["punkt", "punkt_tab", "stopwords"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except (LookupError, IndexError):
            nltk.download(resource, quiet=True)


# Ensure NLTK data is ready
ensure_nltk_resources()

# Initialize stemmer and cached stopwords
_STEMMER = PorterStemmer()
try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Remove HTML tags, URLs, numbers, and punctuation from text.

    Args:
        text (str): Raw input email or message text.

    Returns:
        str: Cleaned text without markup, URLs, numbers, or punctuation.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # 5. Remove numbers and digits
    text = re.sub(r"\d+", " ", text)

    # 6. Remove punctuation
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)

    # 7. Normalize extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text: str) -> str:
    """Complete preprocessing pipeline for an email string.

    Steps:
    1. Lowercase, strip URLs, HTML, digits, punctuation.
    2. Tokenize using NLTK word_tokenize.
    3. Filter out English stopwords.
    4. Apply Porter Stemmer on tokens.
    5. Re-join tokens into a single space-separated string.

    Args:
        text (str): Raw input text.

    Returns:
        str: Preprocessed, stemmed text ready for vectorization.
    """
    cleaned = clean_text(text)

    if not cleaned:
        return ""

    try:
        tokens: List[str] = word_tokenize(cleaned)
    except Exception:
        # Fallback if tokenizer fails
        tokens = cleaned.split()

    filtered_tokens = [
        _STEMMER.stem(word)
        for word in tokens
        if word not in _STOPWORDS and len(word) > 1
    ]

    return " ".join(filtered_tokens)
