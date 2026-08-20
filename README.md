# AI Healthcare Screening Project

An AI-assisted healthcare screening web app built with Flask, MongoDB, and lightning-fast cloud AI. It supports multilingual symptom checking, disease explanations, PDF report generation, doctor lookup, and multimodal image-based medical analysis.

## ⚠️ Disclaimer
> **This application is a prototype and proof-of-concept.** The AI-generated diagnoses, image analyses, and medical reports are **not 100% accurate** and should **never** be used for self-diagnosis or real medical treatment. This project is designed as a foundational prototype that could eventually be built out as an assistant tool for qualified doctors and medical professionals in the future.

## Problem Statement
Healthcare users often need a fast first-pass screening tool that can collect symptoms, explain possible conditions, and present the result in a simple multilingual format. This project addresses that need with an AI-powered symptom checker backed by structured medical reference data, cloud AI text/vision inference, and a MongoDB patient workflow.

## Features
- **Multilingual Symptom Checker:** Select symptoms and extract medical terms seamlessly.
- **Disease Prediction:** Fast ML predictions with confidence scoring.
- **Lightning-Fast AI Explanations:** Powered by Groq for ultra-low latency text generation.
- **Medical Image Analysis:** Powered by Google Gemini Vision API for high-accuracy visual screening.
- **PDF Medical Reports:** Automatically generate downloadable patient reports.
- **Patient History:** Securely stored in MongoDB.
- **Doctor Recommendations:** Based on disease and location.
- **Vercel Ready:** Fully configured for serverless deployment on Vercel.

## Tech Stack
- **Backend:** Flask, PyMongo, Werkzeug
- **AI (Text):** Groq API (`llama-3.1-8b-instant`)
- **AI (Vision):** Google Gemini API (`gemini-1.5-flash`)
- **Machine Learning:** scikit-learn, joblib, pandas, numpy
- **PDF Generation:** reportlab
- **Translation:** deep-translator
- **Frontend:** HTML, CSS, Vanilla JavaScript

## Setup & Local Development

This project uses a hybrid AI architecture to maximize speed and minimize costs, utilizing Groq for text and Gemini for image analysis.

### 1. Prerequisites
- Python 3.11+
- A MongoDB Cluster (MongoDB Atlas)
- A [Groq API Key](https://console.groq.com/keys) (Free)
- A [Google Gemini API Key](https://aistudio.google.com/) (Free)

### 2. Environment Setup
Create a virtual environment and install the minimal dependencies:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration
Rename `.env.example` to `.env` and fill in your keys:
```env
MONGO_URI=your_mongodb_connection_string
AI_PROVIDER=groq
AI_MODEL=llama-3.1-8b-instant
API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run Locally
```bash
python app.py
```
The app will start at `http://127.0.0.1:5000`.

## Vercel Deployment
This project is configured out-of-the-box for Vercel deployment. 
1. Push your code to GitHub.
2. Import the repository into Vercel.
3. In the Vercel Dashboard, go to **Settings > Environment Variables** and add all the variables from your `.env` file.
4. Deploy! Vercel will automatically install the packages in `requirements.txt` and launch the Flask app serverlessly.

## Model Evaluation
The current saved disease prediction model (`disease_model.pkl`) was trained with an 80/20 train/test split.
- **Model:** Multinomial Naive Bayes pipeline
- **Test accuracy:** 0.8947
- **Weighted F1-score:** 0.8944

*(If the symptom dataset changes, run `python train_model.py` to regenerate the `.pkl` files).*

## Notes
- `.env` is ignored by git and should stay local to protect your API keys.
- **Legacy Support:** The application retains fallback logic for Ollama, OpenAI, and Anthropic in the `app.py` source code, but the Vercel deployment strictly uses Groq + Gemini for optimized performance.
