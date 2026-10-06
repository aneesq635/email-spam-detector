"""Feature extraction module using Bag of Words (CountVectorizer)."""

from typing import Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def get_vectorizer(max_features: int = 5000) -> CountVectorizer:
    """Instantiate a CountVectorizer configured for Bag of Words feature extraction.

    Args:
        max_features (int): Maximum vocabulary size. Defaults to 5000.

    Returns:
        CountVectorizer: Unfitted CountVectorizer instance.
    """
    return CountVectorizer(max_features=max_features)


def extract_features(
    corpus: Any,
    vectorizer: CountVectorizer,
    fit: bool = False
) -> Tuple[Any, CountVectorizer]:
    """Transform text corpus into a Bag-of-Words count matrix.

    Args:
        corpus: Iterable of preprocessed text strings.
        vectorizer (CountVectorizer): CountVectorizer instance.
        fit (bool): If True, fit vectorizer on corpus before transform.

    Returns:
        Tuple[Any, CountVectorizer]: (transformed feature matrix, vectorizer)
    """
    if fit:
        features = vectorizer.fit_transform(corpus)
    else:
        features = vectorizer.transform(corpus)

    return features, vectorizer
