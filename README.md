# AI Healthcare Project

An AI-assisted healthcare screening web app built with Flask, MongoDB, and pluggable AI providers. It supports multilingual symptom checking, disease explanations, PDF report generation, doctor lookup, and image-based analysis.

## Problem Statement
Healthcare users often need a fast first-pass screening tool that can collect symptoms, explain possible conditions, and present the result in a simple multilingual format. This project addresses that need with an AI-powered symptom checker backed by structured medical reference data and a MongoDB patient workflow.

## Features
- Multilingual symptom selection and AI-backed symptom extraction.
- Disease prediction with confidence scoring and follow-up questioning.
- Provider-backed disease explanations with cloud and offline fallback support.
- PDF medical report generation.
- Image analysis for uploaded medical visuals, with offline fallback summaries.
- Patient history stored in MongoDB.
- Doctor recommendations based on disease and location.
- CSV fallback mode when MongoDB is unavailable.

## Tech Stack
- Backend: Flask, PyMongo, python-dotenv, Werkzeug
- AI: Ollama (local), OpenAI, Anthropic, Google Gemini, scikit-learn, joblib
- Data: pandas, numpy, CSV/JSON datasets
- PDF: reportlab
- Translation: deep-translator
- Frontend: HTML, CSS, JavaScript

## Model Evaluation
The current saved model was trained with an 80/20 train/test split and saved with joblib compression.

- Model: Multinomial Naive Bayes pipeline
- Test accuracy: 0.8947
- Weighted F1-score: 0.8944
- Saved model size: 22,452 bytes
- Confusion matrix: saved to `confusion_matrix.csv`

## Setup

### Option A: Local AI with Ollama (recommended – no billing, no API key)

1. **Install Ollama** from [ollama.com](https://ollama.com).

2. **Pull the required models**:
   ```bash
   ollama pull llama3.2
   ollama pull llava        # for image analysis (optional)
   ```

3. **Set your `.env`**:
   ```env
   AI_PROVIDER=ollama
   AI_MODEL=llama3.2
   MONGO_URI=your_mongodb_connection_string
   ```

   Optional overrides (these have sensible defaults):
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_VISION_MODEL=llava
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

6. **Run the app**:
   ```bash
   python flask_app.py
   ```

### Option B: Cloud AI Provider

Set `.env` with one of the supported cloud providers:

```env
MONGO_URI=your_mongodb_connection_string
API_KEY=your_api_key_here
AI_PROVIDER=openai        # or: gemini, anthropic
AI_MODEL=gpt-4o-mini      # or: gemini-2.0-flash, claude-3-5-sonnet-latest
```

Supported `AI_PROVIDER` values:
- `ollama` – local Ollama instance (no API key needed)
- `openai`
- `anthropic`
- `gemini`
- `offline` – no AI, uses hardcoded fallback responses

### Option C: Fully Offline (no AI at all)

```env
AI_PROVIDER=offline
AI_MODEL=offline-local
```

### Model Artifacts

Ensure the following files exist in the project root:
- `disease_model.pkl`
- `label_encoder.pkl`

Re-train the model when the symptom dataset changes:
```bash
python train_model.py
```

## Run
```bash
python flask_app.py
```
The app starts at `http://0.0.0.0:5000`.

## Screenshots

> Screenshots are pending — they will be added once the UI is finalized.
>
> Key screens: **Login → Dashboard → Symptom Checker → Results → Report PDF → Image Analysis**

## Notes
- `.env` is ignored by git and should stay local.
- `app.py` was removed because it was an unused Streamlit prototype.
- If MongoDB is unavailable, the app falls back to CSV-based behavior for the supported flows.
- If Ollama is not running when `AI_PROVIDER=ollama`, the app prints a clear error at startup and falls back to offline mode.
