# 🤖 ML-AI Hub

A centralized, modular AI dashboard built with **Flask** that integrates multiple NLP and Machine Learning tools into a single, unified web application.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tools](#tools)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Training the ML Models](#training-the-ml-models)
- [Automated Retraining](#automated-retraining)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Compatibility Notes](#compatibility-notes)

---

## Overview

This project is a hands-on learning platform for Natural Language Processing (NLP) and Machine Learning. Each tool lives in its own module but is served from a single Flask process, sharing a unified navigation hub, consistent API design, and a common service architecture.

---

## Tools

| Tool | Path | Method | Description |
|---|---|---|---|
| 📊 Word Counter | `/wordcounter` | ML (Naive Bayes / TF-IDF) | Counts words and classifies text sentiment |
| 🧩 POS Visualizer | `/posvisualizer` | NLTK PerceptronTagger | Color-coded Part-of-Speech tag visualization |
| 🔑 Keyword Extractor | `/keywordextractor` | Rule-based (NLTK Nouns) | Extracts the most frequent nouns from text |
| 🎯 Intent Detector | `/intentdetector` | Fuzzy Matching (thefuzz) | Maps free text to Tech Support intents via fuzzy string matching |
| 🎓 Admission Chatbot | `/admissionchatbot` | Fuzzy Matching (thefuzz) | Answers university admission questions using a FAQ dataset |
| 📖 Fictometer | `/fictometer` | ML (TF-IDF + Logistic Regression) | Classifies text as Fiction or Non-Fiction (trained on Brown Corpus) |
| 🛡️ Spam Classifier | `/spamclassifier` | ML (Logistic Regression & Naive Bayes) | Detects Spam vs. Ham using the SMS Spam Collection dataset |
| 📂 Category Predictor | `/categorypredictor` | Deep Learning (ANN + Word2Vec / ELMo) | Classifies text into Brown Corpus genre categories using neural embeddings |
| 🔗 Word Similarity | `/wordsimilarity` | Deep Learning (ELMo) | Measures contextual similarity of a word across two sentences |

---

## Architecture

The project follows a **Service-Oriented Architecture** where all ML/NLP logic is fully decoupled from the web layer.

```
ML-AI/
├── run.py                  # Application entry point (port 5000)
├── app/                    # Central Flask application
│   ├── __init__.py         # create_app() factory — loads all models at startup
│   ├── routes.py           # All API endpoints and page routes
│   ├── services.py         # CountService (Word Counter)
│   ├── utils.py            # Shared API helpers (success/error responses)
│   └── templates/          # All HTML templates for every tool
│
├── posvisualizer/app/      # POS Tagger service
├── keywordextractor/app/   # Keyword Extractor service
├── intentdetector/app/     # Intent Detector service
├── admissionchatbot/app/   # Admission Chatbot service
├── fictometer/             # Fictometer ML service + training scripts
├── spamclassifier/         # Spam Classifier ML service + training scripts
├── categorypredictor/      # Category Predictor ANN service + training scripts
├── wordsimilarity/app/     # Word Similarity ELMo service
│
└── scripts/
    └── train_models.bat    # Windows Task Scheduler batch file for weekly retraining
```

Key design decisions:
- **Models are pre-loaded at startup** via `create_app()` to eliminate latency on first requests.
- **ELMo models load lazily** (on first request) due to their large size (~350 MB).
- **All API responses** follow a consistent `{ "status": "success"|"error", "data": {...} }` format.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- `pip` package manager

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd ML-AI
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install core dependencies
```bash
pip install flask joblib nltk scikit-learn thefuzz Levenshtein
```

### 4. Install dependencies for specific tools (as needed)
```bash
# Fictometer & Category Predictor
pip install -r fictometer/requirements.txt
pip install -r categorypredictor/requirements.txt

# Spam Classifier
pip install -r spamclassifier/requirements.txt

# Intent Detector & Admission Chatbot
pip install -r intentdetector/requirements.txt
```

---

## Running the App

```bash
python run.py
```

The app will start at **http://localhost:5000** in debug mode.

> **Note:** On first startup, NLTK resources (Brown Corpus, stopwords, averaged perceptron tagger) will be downloaded automatically.

---

## Training the ML Models

Some tools require trained models before they can make predictions. Run the following scripts **once** before starting the server for the first time:

### Fictometer (Fiction vs. Non-Fiction)
```bash
python fictometer/scripts/trainmodel.py
```

### Spam Classifier (Spam vs. Ham)
```bash
python spamclassifier/scripts/trainmodel.py
```
> Downloads the **SMS Spam Collection** dataset (~5,500 real SMS messages) automatically.

### Category Predictor — Word2Vec + ANN
```bash
python categorypredictor/scripts/trainmodel_word2vec.py
```

### Category Predictor — ELMo + ANN
```bash
python categorypredictor/scripts/trainmodel_elmo.py
```
> ⚠️ ELMo requires downloading a ~350 MB model from TensorFlow Hub on first run.

---

## Automated Retraining

A Windows Batch script is provided to retrain all ML models automatically via the native **Windows Task Scheduler**.

### Register the weekly task (run once in PowerShell):
```powershell
SchTasks /Create /SC WEEKLY /D SUN /TN "WeeklyMLTraining" /TR "c:\Users\mjay2\Desktop\Learn\Excercises\ML-AI\scripts\train_models.bat" /ST 02:00
```

This will silently retrain the Fictometer, Spam Classifier, and both Category Predictor models every **Sunday at 2:00 AM**, keeping your models up-to-date.

---

## API Reference

All endpoints accept and return `application/json`.

| Method | Endpoint | Body Parameters | Description |
|---|---|---|---|
| `POST` | `/api/v1/counts` | `text` | Word count + sentiment analysis |
| `POST` | `/api/v1/analyze_pos` | `text` | Part-of-speech tagging |
| `POST` | `/api/v1/extract_keywords` | `text` | Noun-frequency keyword extraction |
| `POST` | `/api/v1/detect_intent` | `text` | Fuzzy intent detection |
| `POST` | `/api/v1/chat_admission` | `text` | Admission chatbot FAQ answer |
| `POST` | `/api/v1/fictometer` | `text` | Fiction vs. Non-Fiction prediction |
| `POST` | `/api/v1/spamclassifier` | `text`, `algorithm` (`lr`\|`nb`) | Spam vs. Ham classification |
| `POST` | `/api/v1/category_predict` | `text`, `embedding` (`word2vec`\|`elmo`) | Brown Corpus genre prediction |
| `POST` | `/api/v1/word_similarity` | `sentence1`, `sentence2`, `word` | ELMo contextual word similarity |

### Example Request
```bash
curl -X POST http://localhost:5000/api/v1/fictometer \
  -H "Content-Type: application/json" \
  -d '{"text": "The dragon soared over the mountains."}'
```

### Example Response
```json
{
  "status": "success",
  "data": {
    "prediction": "Fiction",
    "confidence": 94.7
  }
}
```

---

## Project Structure

```
ML-AI/
├── app/                      # Central Flask hub
│   ├── __init__.py           # Application factory
│   ├── routes.py             # All routes and API endpoints
│   ├── services.py           # Word Counter / Sentiment service
│   ├── utils.py              # API response helpers
│   └── templates/            # HTML templates for all tools
│
├── posvisualizer/app/        # NLTK POS tagging service
├── keywordextractor/app/     # Noun-frequency keyword service
├── intentdetector/app/       # Fuzzy intent matching service
├── admissionchatbot/app/     # FAQ chatbot service
│
├── fictometer/
│   ├── app/services.py       # Loads trained TF-IDF model
│   ├── models/               # Saved .joblib model files
│   └── scripts/trainmodel.py # Brown Corpus training script
│
├── spamclassifier/
│   ├── app/services.py       # Loads both LR and NB models
│   ├── models/               # Saved .joblib model files
│   └── scripts/trainmodel.py # SMS Spam Collection training script
│
├── categorypredictor/
│   ├── app/services.py       # Loads Word2Vec + ELMo ANN models
│   ├── models/               # Saved .keras and .model files
│   └── scripts/
│       ├── preprocess.py             # Shared tokenizer utility
│       ├── trainmodel_word2vec.py    # Word2Vec + ANN training
│       └── trainmodel_elmo.py       # ELMo + ANN training
│
├── wordsimilarity/app/       # ELMo contextual word similarity service
│
├── scripts/
│   └── train_models.bat      # Weekly Windows Task Scheduler script
│
└── run.py                    # App entry point
```

---

## Compatibility Notes

This project requires careful version alignment between Python, TensorFlow, and SetupTools.

- Python **3.10–3.12** recommended (Python 3.13 may have TensorFlow compatibility issues)
- TensorFlow **2.13+** required for ELMo (Word Similarity, Category Predictor)
- If you encounter TensorBoard or SetupTools conflicts, refer to these known issues:
  - https://github.com/tensorflow/tensorboard/issues/7064
  - https://github.com/tensorflow/tensorflow/issues/102890

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
