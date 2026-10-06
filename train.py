"""Training script for Email Spam Detection using Multinomial Naive Bayes.

Pipeline:
1. Load and clean dataset from data/
2. Preprocess text using NLTK pipeline
3. Extract Bag of Words features (CountVectorizer, max_features=5000)
4. Train/test stratified split (80/20)
5. Train MultinomialNB model
6. Evaluate: Accuracy, Precision, Recall, F1-score, and save confusion matrix to outputs/
7. Save model and vectorizer to models/
"""

import os
import pickle
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data_loader import load_dataset
from src.preprocess import preprocess_text
from src.features import get_vectorizer


def train_model() -> None:
    """Execute the end-to-end training, evaluation, and artifact saving pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading dataset from data/ directory...")
    print("=" * 60)
    try:
        df = load_dataset()
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")
        return

    print(f"Dataset successfully loaded. Total samples after deduplication: {len(df)}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()} (0=Ham, 1=Spam)\n")

    print("=" * 60)
    print("STEP 2: Preprocessing text (cleaning, tokenization, stemming)...")
    print("=" * 60)
    df["clean_text"] = df["text"].apply(preprocess_text)
    print("Text preprocessing completed.\n")

    print("=" * 60)
    print("STEP 3: Splitting dataset (80% train, 20% test stratified)...")
    print("=" * 60)
    X = df["clean_text"]
    y = df["label"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}\n")

    print("=" * 60)
    print("STEP 4: Extracting Bag of Words features (CountVectorizer)...")
    print("=" * 60)
    vectorizer = get_vectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features\n")

    print("=" * 60)
    print("STEP 5: Training Multinomial Naive Bayes model...")
    print("=" * 60)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Model training completed successfully.\n")

    print("=" * 60)
    print("STEP 6: Evaluating model performance...")
    print("=" * 60)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (Normal)", "Spam"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (Ham)", "Spam"],
        yticklabels=["Normal (Ham)", "Spam"],
    )
    plt.title("Confusion Matrix - Email Spam Detector")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}\n")

    print("=" * 60)
    print("STEP 7: Saving model & vectorizer pickles...")
    print("=" * 60)
    model_path = os.path.join(models_dir, "model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # Save metrics metadata for the Streamlit sidebar
    metrics_path = os.path.join(models_dir, "metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(
            {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "train_samples": len(X_train_raw),
                "test_samples": len(X_test_raw),
                "vocab_size": len(vectorizer.vocabulary_),
            },
            f,
        )

    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\n[SUCCESS] Pipeline completed successfully!")


if __name__ == "__main__":
    train_model()
