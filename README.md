# 🛡️ Email Spam Detection System (NLP Assignment)

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify spam emails and messages. Built with Python 3, scikit-learn, NLTK, and Streamlit.

---

## 📌 Project Overview
Spam emails clutter inboxes and expose users to phishing and security threats. This application provides a complete NLP-driven pipeline that:
1. Automatically loads and standardizes any spam dataset CSV.
2. Cleans, tokenizes, removes stopwords, and stems email text.
3. Extracts vocabulary counts using Bag of Words (BoW).
4. Trains a Multinomial Naive Bayes classifier.
5. Deploys an interactive Streamlit web dashboard for real-time testing.

---

## ✨ Features
- **Auto-Detection of Columns**: Automatically recognizes text columns (`text`, `message`, `v2`, `content`, `sms`, etc.) and target label columns (`label`, `v1`, `label_num`, `category`, etc.).
- **Automatic Label Normalization**: Maps any binary target values (`spam`/`ham`, `1`/`0`, `junk`/`normal`) to `spam = 1` and `ham = 0`.
- **Deduplication & Null Handling**: Cleans raw data by dropping missing values and duplicate messages.
- **Robust NLP Preprocessing**:
  - Lowercasing
  - Stripping HTML tags, URLs, and email addresses
  - Stripping numbers and punctuation
  - NLTK Tokenization with auto-download of required resources (`punkt`, `stopwords`)
  - Stopword filtering and Porter Stemming
- **Bag of Words Feature Representation**: CountVectorizer with top 5,000 features.
- **Multinomial Naive Bayes Classifier**: Fast, probabilistic text classification with stratified 80/20 train/test evaluation.
- **Performance Evaluation**: Logs accuracy, precision, recall, F1-score, and exports a high-resolution Confusion Matrix plot to `outputs/confusion_matrix.png`.
- **Pickle Artifact Deployment**: Saves model, vectorizer, and performance metrics to `models/`.
- **Streamlit Web Application**: Clean interface featuring sample email presets, spam probability confidence meters, and preprocessed token inspection.

---

## 🛠️ Tech Stack
- **Language**: Python 3.13+
- **NLP / Text Processing**: NLTK (`PorterStemmer`, `word_tokenize`, `stopwords`), Regex
- **Machine Learning**: scikit-learn (`MultinomialNB`, `CountVectorizer`, `train_test_split`, metrics)
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Interface**: Streamlit
- **Model Serialization**: Python `pickle`

---

## 📁 Project Structure
```text
spam-detector/
│
├── data/                    # Place your dataset CSV file here
├── models/                  # Serialized pickle files (model.pkl, vectorizer.pkl, metrics.pkl)
├── outputs/                 # Saved evaluation plots (confusion_matrix.png)
├── src/                     # Core NLP pipeline modules
│   ├── __init__.py
│   ├── data_loader.py       # Auto-detects columns, normalizes labels, drops duplicates & nulls
│   ├── preprocess.py        # Cleans markup/URLs/digits, tokenizes, removes stopwords, stems
│   ├── features.py          # Bag of Words CountVectorizer extraction
│   └── predict.py           # Inference function predict_email() loading pickles
├── train.py                 # End-to-end model training, evaluation & artifact saving
├── app.py                   # Streamlit interactive web interface
├── requirements.txt         # Project dependencies
└── README.md                # Documentation and execution guide
```

---

## 🧠 How the Pipeline Works: Bag of Words (BoW) Explained Simply

1. **Text Preprocessing**:
   - An email like `"URGENT! You've won 500 dollars. Click http://spam.link"` becomes:
   - Lowercased: `"urgent! you've won 500 dollars. click http://spam.link"`
   - Cleaned (no URLs, numbers, punctuation): `"urgent you ve won dollars click"`
   - Tokenized & Stemmed: `["urgent", "won", "dollar", "click"]` (stopwords like "you", "ve" removed).

2. **Bag of Words (BoW)**:
   - Bag of Words converts freeform text into fixed numerical vectors.
   - It builds a vocabulary dictionary of the most frequent 5,000 words across the entire training corpus.
   - For any message, it counts how many times each dictionary word appears, ignoring sentence grammar or word order (like throwing words into a bag and counting them).
   - Example Vector: `[urgent: 1, prize: 0, dollar: 1, click: 1, meeting: 0, ...]`

3. **Multinomial Naive Bayes (MNB)**:
   - MNB applies Bayes' theorem to calculate the probability:
     $$P(\text{Spam} \mid \text{Words}) = \frac{P(\text{Words} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{Words})}$$
   - Because spam messages disproportionately contain words like *"prize"*, *"win"*, *"free"*, and *"urgent"*, the model calculates higher probability for the spam class.

---

## 🚀 Setup Instructions

### 1. Clone / Navigate to Project Directory
```bash
cd c:\Users\don33\Documents\taha-assignment
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

**On Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Training the Model

1. **Add Dataset**: Place your dataset CSV file inside the `data/` folder (e.g., `data/spam.csv` or `data/emails.csv`).
2. **Execute Training**:
   ```bash
   python train.py
   ```
3. **What happens during training**:
   - Dataset is auto-detected, deduplicated, and normalized.
   - NLTK downloads necessary tokenization resources quietly.
   - Data is split 80% train / 20% test (stratified by class).
   - Features are extracted with `CountVectorizer(max_features=5000)`.
   - Naive Bayes is trained and evaluated.
   - Metrics are printed in the terminal (Accuracy, Precision, Recall, F1).
   - `outputs/confusion_matrix.png` is generated.
   - Model and vectorizer are saved to `models/model.pkl` and `models/vectorizer.pkl`.

---

## 💻 Running the Streamlit App

Launch the application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ✉️ Sample Emails to Test

### 🔴 Sample Spam Email:
```text
Congratulations! You have been selected as the winner of the global lottery $1,000,000 prize. Reply immediately with your bank account details and ID number to claim your cash reward today!
```
- **Expected Result**: `🚨 SPAM DETECTED` (Confidence > 95%)

### 🟢 Sample Normal (Ham) Email:
```text
Hi Sarah, please find attached the presentation slides for our engineering sync tomorrow at 10 AM. Let me know if you'd like to adjust any slides before the meeting.
```
- **Expected Result**: `✅ NORMAL (Not Spam)` (Confidence > 95%)

---

## 📈 Model Evaluation & Real Dataset Results
Trained on Enron email spam collection (`spam_ham_dataset.csv`, 4,993 cleaned emails after deduplication):
- **Accuracy**: **95.30%**
- **Precision (Spam)**: **88.68%**
- **Recall (Spam)**: **96.25%**
- **F1-Score (Spam)**: **92.31%**
- **Normal (Ham) Precision/Recall**: 98% / 95%
- **Vocabulary Size**: 5,000 Bag-of-Words features
- **Confusion Matrix**: Saved to `outputs/confusion_matrix.png`
