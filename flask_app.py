import os
import requests as http_requests  # renamed to avoid clash with flask.request
import secrets
import pandas as pd
import numpy as np
import joblib
import json
import ast
import base64
from io import BytesIO
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file, flash

from deep_translator import GoogleTranslator
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
from pymongo import MongoClient
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import certifi
from datetime import timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Import local modules
from medical_data import get_medical_info
from doctor_service import find_doctors_for_disease

# Register Unicode Font for Hindi/Gujarati support
# Nirmala UI is a standard Windows font that covers Indic scripts well
FONT_PATH = "C:\\Windows\\Fonts\\Nirmala.ttc"
if os.path.exists(FONT_PATH):
    try:
        # Nirmala.ttc is a TrueType Collection, usually index 0 is regular
        pdfmetrics.registerFont(TTFont('Nirmala', FONT_PATH))
        USING_UNICODE_FONT = True
    except Exception as fe:
        print(f"Font Registration Error: {fe}")
        USING_UNICODE_FONT = False
else:
    print("Nirmala UI font not found. Non-English PDFs may show boxes.")
    USING_UNICODE_FONT = False

app = Flask(__name__)
app.secret_key = "healthcare_app_secure_key_2026"
# Session expires when browser closes, but stays active while user is working
app.config['SESSION_PERMANENT'] = False  # This makes session expire on browser close
# Note: Don't set PERMANENT_SESSION_LIFETIME when SESSION_PERMANENT is False

# MongoDB Config with Fallback
try:
    # MongoDB Atlas Connection String
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set")
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db = mongo_client["healthcare_db"]
    # Check connection
    mongo_client.server_info()
    users_col = db["users"]
    patients_col = db["patients"]
    history_col = db["medical_history"]
    predictions_col = db["predictions"] # New collection for patient history table
    USING_MONGODB = True
    MONGO_ERROR = None
    print("Connected to MongoDB")
except Exception as e:
    print(f"MongoDB not detected (Using CSV Fallback): {e}")
    USING_MONGODB = False
    MONGO_ERROR = str(e)
    users_col = None
    patients_col = None
    history_col = None
    predictions_col = None

# --- AI CONFIG ---
AI_PROVIDER = os.getenv("AI_PROVIDER", os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

AI_MODEL_NAME = os.getenv("AI_MODEL", "").strip()

# --- OLLAMA CONFIG ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")


def _offline_text_response(prompt):
    prompt_lower = prompt.lower()

    if "provide a comprehensive, detailed medical explanation" in prompt_lower:
        disease = prompt.split("for '")[-1].split("'")[0] if "for '" in prompt else "the condition"
        return f"""⚠️ [OFFLINE MODE]

**Disease:** {disease}

**1. OVERVIEW:**
{disease} is a medical condition that can require attention and follow-up depending on severity.

**2. CAUSES:**
• Infection or inflammation
• Lifestyle-related factors
• Genetic predisposition
• Other underlying health issues

**3. SYMPTOMS:**
• Pain or discomfort
• Fatigue or weakness
• Fever or inflammation
• Condition-specific symptoms

**4. DIAGNOSIS:**
A doctor usually confirms this through symptoms, examination, and tests if needed.

**5. TREATMENT:**
• Rest and hydration
• Doctor-prescribed medicines
• Monitoring symptoms
• Follow-up if symptoms worsen

**6. PREVENTION:**
• Maintain hygiene
• Eat a balanced diet
• Stay hydrated
• Follow medical advice early

**7. WHEN TO SEE A DOCTOR:**
• Symptoms get worse
• High fever persists
• Severe pain starts
• New worrying signs appear

**8. LIFESTYLE RECOMMENDATIONS:**
• Rest adequately
• Avoid stress where possible
• Keep a simple diet
• Seek help if symptoms do not improve
""", None

    if "write a professional medical screening report" in prompt_lower:
        disease = prompt.split("for '")[-1].split("'")[0] if "for '" in prompt else "the condition"
        return f"""MEDICAL SCREENING REPORT: {disease}

FINDINGS:
- Screening indicates symptoms consistent with {disease}.

ADVICE:
- Rest and monitor symptoms.
- Follow medical guidance.
- Consult a clinician if symptoms worsen.

NOTE:
- This is an offline generated report and not a substitute for clinical diagnosis.
""", None

    if "extract only the symptoms mentioned" in prompt_lower or "extracted symptoms" in prompt_lower:
        return "[]", None

    return "Offline mode: a local response could not be generated for this prompt.", None


def _offline_image_response(prompt, image_bytes):
    try:
        from PIL import ImageStat
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / 3
        width, height = image.size

        return f"""⚠️ [OFFLINE MODE] Image Analysis (local heuristic)

Reason: No external AI provider is available.

Image details:
- Size: {width} x {height}
- Average brightness: {brightness:.1f}

    Suggested interpretation:
    - The upload looks like a document or scan.
    - No clinical AI interpretation was performed offline.
    - For true image understanding, use a local vision model or a cloud provider with image support.
""", None
    except Exception as e:
        return None, str(e)

def _load_ai_client():
    provider = AI_PROVIDER

    if provider == "offline":
        return {
            "provider": provider,
            "model": AI_MODEL_NAME or "offline-local",
            "client": None,
        }

    if provider == "ollama":
        model_name = AI_MODEL_NAME or "llama3.2"
        # Quick connectivity check
        try:
            http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {OLLAMA_BASE_URL}. "
                "Make sure Ollama is running (ollama serve)."
            ) from exc
        return {
            "provider": "ollama",
            "model": model_name,
            "client": None,  # we use requests directly
        }

    if provider == "groq":
        try:
            from groq import Groq
        except Exception as exc:
            raise RuntimeError("Groq provider selected but the groq package is not installed.") from exc
        if not API_KEY:
            raise RuntimeError("API_KEY is not set for Groq.")
        model_name = AI_MODEL_NAME or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return {
            "provider": provider,
            "model": model_name,
            "client": Groq(api_key=API_KEY),
        }

    if provider == "openai":
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OpenAI provider selected but the openai package is not installed.") from exc
        if not API_KEY:
            raise RuntimeError("API_KEY is not set for OpenAI.")
        model_name = AI_MODEL_NAME or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return {
            "provider": provider,
            "model": model_name,
            "client": OpenAI(api_key=API_KEY),
        }

    if provider == "anthropic":
        try:
            import anthropic
        except Exception as exc:
            raise RuntimeError("Anthropic provider selected but the anthropic package is not installed.") from exc
        if not API_KEY:
            raise RuntimeError("API_KEY is not set for Anthropic.")
        model_name = AI_MODEL_NAME or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        return {
            "provider": provider,
            "model": model_name,
            "client": anthropic.Anthropic(api_key=API_KEY),
        }

    try:
        import google.generativeai as genai
        from google.api_core.exceptions import ResourceExhausted, PermissionDenied, NotFound
    except Exception as exc:
        raise RuntimeError("Gemini provider selected but google-generativeai is not available.") from exc

    if not API_KEY:
        raise RuntimeError("API_KEY is not set for Gemini.")
    genai.configure(api_key=API_KEY)
    model_name = AI_MODEL_NAME or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return {
        "provider": "gemini",
        "model": model_name,
        "client": genai.GenerativeModel(model_name),
        "errors": {
            "ResourceExhausted": ResourceExhausted,
            "PermissionDenied": PermissionDenied,
            "NotFound": NotFound,
        },
    }


AI_CLIENT = None
AI_MODEL_NAME_RESOLVED = None
try:
    AI_CLIENT = _load_ai_client()
    AI_MODEL_NAME_RESOLVED = AI_CLIENT["model"]
    print(f"AI Provider Initialized: {AI_CLIENT['provider']} ({AI_MODEL_NAME_RESOLVED})")
except Exception as e:
    AI_CLIENT = None
    print(f"AI provider unavailable: {e}")


def _ai_error_message(error_text):
    text = str(error_text).lower()
    if "quota" in text or "billing" in text:
        return "AI quota exceeded or billing is not enabled for this project."
    if "reported as leaked" in text:
        return "The API key was reported as leaked. Replace it with a new key."
    if "api key" in text and ("invalid" in text or "not valid" in text or "permission" in text):
        return "The API key is invalid or lacks permission. Check API_KEY in .env."
    if "not found" in text or "unsupported" in text:
        return f"The selected model '{AI_MODEL_NAME_RESOLVED}' is not available for this provider."
    return str(error_text)


def call_ai_text(prompt, **kwargs):
    if not AI_CLIENT:
        return None, "AI provider is not available. Check API_KEY and AI_PROVIDER in .env."

    provider = AI_CLIENT["provider"]
    try:
        if provider == "offline":
            return _offline_text_response(prompt)

        if provider == "ollama":
            resp = http_requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": AI_CLIENT["model"], "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", ""), None

        if provider == "groq":
            client = AI_CLIENT["client"]
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=AI_CLIENT["model"],
                **kwargs,
            )
            return chat_completion.choices[0].message.content, None

        if provider == "openai":
            client = AI_CLIENT["client"]
            response = client.responses.create(
                model=AI_CLIENT["model"],
                input=prompt,
                **kwargs,
            )
            return response.output_text, None

        if provider == "anthropic":
            client = AI_CLIENT["client"]
            response = client.messages.create(
                model=AI_CLIENT["model"],
                max_tokens=kwargs.pop("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            text = "".join(part.text for part in response.content if getattr(part, "type", None) == "text")
            return text, None

        response = AI_CLIENT["client"].generate_content(prompt, **kwargs)
        return response.text, None

    except Exception as e:
        return None, _ai_error_message(e)


def call_ai_image(prompt, image_bytes, mime_type="image/png"):
    if not AI_CLIENT:
        return None, "AI provider is not available. Check API_KEY and AI_PROVIDER in .env."

    provider = AI_CLIENT["provider"]
    try:
        if provider == "offline":
            return _offline_image_response(prompt, image_bytes)

        if provider == "ollama":
            # Resize image to avoid Ollama memory issues with large uploads
            try:
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                max_dim = 1024
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim))
                buf = BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
            except Exception:
                pass  # send original bytes if resize fails

            encoded = base64.b64encode(image_bytes).decode("utf-8")
            resp = http_requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [encoded],
                        }
                    ],
                    "stream": False,
                },
                timeout=300,
            )
            if resp.status_code != 200:
                err_detail = resp.text[:300] if resp.text else resp.reason
                return None, f"Ollama vision error ({resp.status_code}): {err_detail}"
            msg = resp.json().get("message", {})
            return msg.get("content", ""), None

        if provider == "groq":
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{encoded}"
            client = AI_CLIENT["client"]
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                model="llama-3.2-11b-vision-preview",
            )
            return chat_completion.choices[0].message.content, None

        if provider == "openai":
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{encoded}"
            client = AI_CLIENT["client"]
            response = client.responses.create(
                model=AI_CLIENT["model"],
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
            )
            return response.output_text, None

        if provider == "anthropic":
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            client = AI_CLIENT["client"]
            response = client.messages.create(
                model=AI_CLIENT["model"],
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded}},
                        ],
                    }
                ],
            )
            text = "".join(part.text for part in response.content if getattr(part, "type", None) == "text")
            return text, None

        image = Image.open(BytesIO(image_bytes))
        response = AI_CLIENT["client"].generate_content([prompt, image])
        return response.text, None

    except Exception as e:
        return None, _ai_error_message(e)

# --- PDF CONFIG ---
USING_UNICODE_FONT = False # Set to True if Nirmala or other Indic font is installed

# --- LOAD MODELS ---
try:
    model = joblib.load("disease_model.pkl")
    le = joblib.load("label_encoder.pkl")
    df_data = pd.read_csv("data/Diseases_and_Symptoms_data.csv")
    symptoms_list = [s.strip() for s in df_data.drop("diseases", axis=1).columns.tolist()]
    
    # Load additional data for report/dictionary
    desc = pd.read_csv("data/description.csv")
    prec = pd.read_csv("data/precautions.csv")
    meds = pd.read_csv("data/medications.csv")
    diet = pd.read_csv("data/diets.csv")
    work = pd.read_csv("data/workout.csv")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    symptoms_list = []

# --- LOAD TRANSLATIONS ---
def load_translations():
    try:
        with open("data/symptom_translations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}
TRANSLATIONS = load_translations()

# Create Reverse Map for Output (English -> Local)
REVERSE_TRANSLATIONS = {}
for lang, mapping in TRANSLATIONS.items():
    REVERSE_TRANSLATIONS[lang] = {v: k for k, v in mapping.items()}

# --- REPORT HEADERS ---
REPORT_HEADERS = {
    "English": {
        "title": "MEDICAL SCREENING REPORT",
        "meds": "RECOMMENDED MEDICATIONS & TREATMENT:",
        "prec": "GENERAL PRECAUTIONS:",
        "diet": "DIET PLAN:",
        "work": "LIFESTYLE & EXERCISE:",
        "disclaimer": "Disclaimer: This is an AI-generated screening report. It is not an official medical diagnosis. Please consult a qualified doctor for clinical verification."
    },
    "Hindi": {
        "title": "मेडिकल स्क्रीनिंग रिपोर्ट",
        "meds": "अनुशंसित दवाएं और उपचार:",
        "prec": "सामान्य सावधानियां:",
        "diet": "आहार योजना:",
        "work": "जीवनशैली और व्यायाम:",
        "disclaimer": "अस्वीकरण: यह एक एआई-जेनरेटेड रिपोर्ट है। यह आधिकारिक चिकित्सा निदान नहीं है। नैदानिक सत्यापन के लिए कृपया डॉक्टर से परामर्श लें।"
    },
    "Gujarati": {
        "title": "મેડિકલ સ્ક્રિનિંગ રિપોર્ટ",
        "meds": "ભલામણ કરેલ દવાઓ અને સારવાર:",
        "prec": "સામાન્ય સાવચેતીઓ:",
        "diet": "આહાર યોજના:",
        "work": "જીવનશૈલી અને કસરત:",
        "disclaimer": "ડિસ્ક્લેમર: આ એક AI-જનરેટેડ રિપોર્ટ છે. તે સત્તાવાર તબીબી નિદાન નથી. કૃપા કરીને ડોક્ટરની સલાહ લો."
    }
}

# --- HELPER FUNCTIONS ---

def get_discriminating_symptom(candidates, current_symptoms, asked_symptoms):
    """Finds the symptom that best distinguishes between the candidate diseases."""
    try:
        # Filter data for candidate diseases
        subset = df_data[df_data['diseases'].isin(candidates)]
        
        if subset.empty:
            return None
            
        # Calculate symptom frequencies for these diseases
        symptom_stats = subset.drop('diseases', axis=1).mean()
        
        best_symptom = None
        max_variance = -1
        
        for symptom, freq in symptom_stats.items():
            # Skip if already known or asked
            if symptom in current_symptoms or symptom in asked_symptoms:
                continue
                
            # Check variance across candidates
            disease_profiles = subset.groupby('diseases')[symptom].mean()
            variance = disease_profiles.var()
            
            if pd.isna(variance): variance = 0
            
            if variance > max_variance:
                max_variance = variance
                best_symptom = symptom
                
        return best_symptom
    except Exception as e:
        print(f"Logic Error: {e}")
        return None



# --- AUTH DECORATORS ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            # If it's an API call, return JSON instead of redirect
            if request.path.startswith('/api/'):
                return jsonify({"error": "Session expired. Please login again.", "status": "unauthorized"}), 401
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session or session['user'].get('role') != role:
                return jsonify({"error": "Forbidden: Restricted to " + role}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- DATA MERGING HELPER ---
def get_detailed_info(disease):
    d_lower = disease.strip().lower()
    
    # 1. Start with hardcoded metadata (Severity, Specialist)
    info = get_medical_info(disease)
    
    # 2. Add Description
    try:
        match = desc[desc['Disease'].str.strip().str.lower() == d_lower]
        if not match.empty:
            info['description'] = match.iloc[0]['Description']
    except: pass
    
    # 3. Add Precautions (as list)
    try:
        match = prec[prec['Disease'].str.strip().str.lower() == d_lower]
        if not match.empty:
            row = match.iloc[0]
            # Get all columns except 'Disease'
            p_list = [str(val) for col, val in row.items() if 'Precaution' in col and pd.notna(val)]
            info['precautions'] = p_list
    except: pass
    
    # 4. Add Medications (parse string representation of list)
    try:
        match = meds[meds['Disease'].str.strip().str.lower() == d_lower]
        if not match.empty:
            med_str = match.iloc[0]['Medication']
            # Safely evaluate string like "['drug1', 'drug2']"
            info['medications'] = ast.literal_eval(med_str)
    except: pass
    
    # 5. Add Diets (parse string representation of list)
    try:
        match = diet[diet['Disease'].str.strip().str.lower() == d_lower]
        if not match.empty:
            diet_str = match.iloc[0]['Diet']
            info['diets'] = ast.literal_eval(diet_str)
    except: pass
    
    # 6. Add Workouts (parse string representation of list)
    try:
        match = work[work['Disease'].str.strip().str.lower() == d_lower]
        if not match.empty:
            work_str = match.iloc[0]['Workout']
            info['workouts'] = ast.literal_eval(work_str)
    except: pass
    
    return info

def _language_register_instruction(language):
    """Returns a tone/register instruction for the given language."""
    if language == "Gujarati":
        return (
            "Write in everyday spoken Gujarati as used in Saurashtra/Kathiyawad. "
            "Avoid formal literary or heavily Sanskrit-influenced Gujarati. "
            "Use a conversational tone, like a doctor talking to a patient."
        )
    if language == "Hindi":
        return (
            "Write in simple everyday spoken Hindi (Hindustani). "
            "Avoid shudh or heavily Sanskritized formal Hindi. "
            "Use a conversational tone, like a doctor talking to a patient."
        )
    return f"Use {language} throughout."


def translate_info(info, language):
    """Uses AI to translate medical info to target language."""
    if language.lower() == 'english' or language.startswith('en'):
        return info
    
    register = _language_register_instruction(language)
    try:
        prompt = (
            f"Translate the following medical information into simple {language}. "
            f"{register} "
            "Maintain the JSON structure. Translate the values, not the keys. "
            f"Target Language: {language}\n"
            f"Data: {json.dumps(info)}"
        )
        text, ai_error = call_ai_text(prompt)
        if ai_error:
            print(f"Translation Error: {ai_error}")
            return info
        text = text.strip()
        if "```" in text:
            text = text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return info

# --- PREDICTION HISTORY HELPER ---
def save_to_prediction_history(disease, confidence, symptoms):
    if USING_MONGODB and 'user' in session:
        try:
            predictions_col.insert_one({
                "username": session["user"]["username"],
                "timestamp": datetime.datetime.now(),
                "symptoms": symptoms,
                "disease": disease,
                "confidence": round(float(confidence) * 100, 2), # Store as percentage
                "age": "25", # Placeholder or could be added to session
                "gender": "Male" # Placeholder
            })
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
    return False
    return decorator

@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    """
    Handles the full diagnostic flow:
    1. Initial Prediction based on symptoms.
    2. Refinement if confidence is low (returns a question).
    3. Final result if confidence is high or specific flag set.
    """
    data = request.json
    current_symptoms = data.get("symptoms", [])
    asked_symptoms = data.get("asked_symptoms", [])
    force_final = data.get("force_final", False)
    language_code = data.get("language", "en-IN")
    
    # Map code to Name
    lang_name = "English"
    if language_code.startswith("hi"): lang_name = "Hindi"
    elif language_code.startswith("gu"): lang_name = "Gujarati"
    
    if not model:
        return jsonify({"status": "error", "message": "Model not loaded"})

    # Prepare input vector with feature names so it matches the trained model
    input_vector = pd.DataFrame([
        [1 if s in current_symptoms else 0 for s in symptoms_list]
    ], columns=symptoms_list)
    
    # Predict Probabilities
    probs = model.predict_proba(input_vector)[0]
    
    # Get Top Candidates
    top_indices = np.argsort(probs)[-3:][::-1]
    candidates = []
    candidate_names = []
    
    for idx in top_indices:
        prob = probs[idx]
        if prob > 0.05: # Filter noise
            disease_name = le.inverse_transform([idx])[0]
            candidates.append({"disease": disease_name, "prob": float(prob)})
            candidate_names.append(disease_name)
    
    if not candidates:
        return jsonify({
            "status": "final",
            "result": {
                "disease": "Unknown",
                "confidence": 0.0,
                "info": get_medical_info("default")
            }
        })

    top_prob = candidates[0]['prob']
    
    # Decision Logic
    # If high confidence OR force_final OR only one candidate OR no more questions
    if top_prob > 0.9 or force_final or len(candidates) == 1:
        disease = candidates[0]['disease']
        info = get_detailed_info(disease)
        
        # Localize Results
        localized_info = translate_info(info, lang_name)
            
        return jsonify({
            "status": "final",
            "result": {
                "disease": disease,
                "confidence": float(top_prob),
                "info": localized_info
            },
            "history_saved": save_to_prediction_history(disease, top_prob, current_symptoms)
        })
    else:
        # Refinement Phase: Find next question
        next_symptom = get_discriminating_symptom(candidate_names, current_symptoms, asked_symptoms)
        
        if next_symptom:
            # --- TRANSLATE QUESTION ---
            lang_key = language_code.split("-")[0]
            question_text = f"Do you also experience {next_symptom}?" # Default English
            
            # Localize symptom name
            localized_symptom = next_symptom
            if lang_key in REVERSE_TRANSLATIONS and next_symptom in REVERSE_TRANSLATIONS[lang_key]:
                localized_symptom = REVERSE_TRANSLATIONS[lang_key][next_symptom]
            
            # Localize Question Template
            if lang_key == 'hi':
                question_text = f"क्या आपको {localized_symptom} भी महसूस हो रहा है?"
            elif lang_key == 'gu':
                question_text = f"શું તમને {localized_symptom} પણ થાય છે?"
            else:
                 # Fallback: if symptom is translated but template isn't (minor improvement)
                 question_text = f"Do you also experience {localized_symptom}?"

            return jsonify({
                "status": "question",
                "question_symptom": next_symptom,
                "question_text": question_text, # Send pre-formatted question
                "candidates": candidate_names
            })
        else:
            # No better question found, return top result
            disease = candidates[0]['disease']
            info = get_detailed_info(disease)
            localized_info = translate_info(info, lang_name)

            return jsonify({
                "status": "final",
                "result": {
                    "disease": disease,
                    "confidence": float(top_prob),
                    "info": localized_info
                },
                "history_saved": save_to_prediction_history(disease, top_prob, current_symptoms)
            })


# --- ROUTES ---

@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get("role", "patient")
        
        if not USING_MONGODB:
            return render_template("register.html", error=f"Database connection error. Details: {MONGO_ERROR}")

        if users_col.find_one({"username": username}):
            return render_template("register.html", error="Username already exists")

        hashed_pw = generate_password_hash(password)
        users_col.insert_one({
            "username": username,
            "email": email,
            "password": hashed_pw,
            "role": role,
            "created_at": datetime.datetime.now()
        })

        flash(f"Account for {username} created! Please login.")
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    if not USING_MONGODB:
        return render_template("login.html", error="Database connection error. Please contact administrator.")
    
    user = users_col.find_one({"username": username})
    if not user:
        return render_template("login.html", error="Username not found. Please register.")
    
    if check_password_hash(user["password"], password):
        # Session expires on browser close
        session["user"] = {"username": user["username"], "role": user["role"]}
        return redirect(url_for("dashboard"))
    else:
        return render_template("login.html", error="Incorrect password.")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"], symptoms=symptoms_list)

# --- SYMPTOM EXTRACTION HELPER ---
def local_extract_symptoms(text, lang_code):
    found = set()
    text_lower = text.lower().strip()
    lang_key = lang_code.split("-")[0]
    
    # 1. Direct English Match (Fastest)
    for sym in symptoms_list:
        if sym.lower() in text_lower:
            found.add(sym)
            
    # 2. Translation Dictionary Match
    if lang_key in TRANSLATIONS:
        lang_map = TRANSLATIONS[lang_key]
        
        # Comprehensive Colloquial Map for "Perfect" Detection
        colloquial_map = {
            "hi": {
                "सर में दर्द": "headache", "सर दर्द": "headache", "माथा दर्द": "headache",
                "पेट में दर्द": "stomach pain", "पेट दर्द": "stomach pain",
                "बुखार": "fever", "ताप": "fever",
                "बदन दर्द": "muscle pain", "शरीर दर्द": "muscle pain",
                "सांस फूलना": "shortness of breath", "सांस लेने में दिक्कत": "shortness of breath",
                "ठंड": "chills", "कपकपी": "chills",
                "उल्टी": "vomiting", "जी मिचलाना": "nausea",
                "दस्त": "diarrhea", "पेचिश": "diarrhea",
                "खांसी": "cough", "जुकाम": "common cold", "शर्दी": "common cold", # Note: common cold might be a disease, but used as symptom
                "कमजोरी": "weakness", "थकान": "fatigue",
                "चक्कर": "dizziness", "खुजली": "itching of skin",
                "सूजन": "skin swelling", "जलन": "burning",
                "कब्ज": "constipation"
            },
            "gu": {
                "માથું દુખે": "headache", "માથાનો દુખાવો": "headache",
                "પેટમાં દુખે": "stomach pain", "પેટનો દુખાવો": "stomach pain",
                "તાવ": "fever", "ગરમી": "fever",
                "શરીર દુખે": "muscle pain", "હાડકા દુખે": "muscle pain",
                "શ્વાસ ચડે": "shortness of breath", "દમ": "shortness of breath",
                "ઠંડી": "chills", "ધ્રુજારી": "chills",
                "ઉલટી": "vomiting", "ઓબકા": "nausea",
                "ઝાડા": "diarrhea", "જુલાબ": "diarrhea",
                "ખાંસી": "cough", "ઉધરસ": "cough",
                "શરદી": "cough", "ઝરઝરિયા": "nasal congestion",
                "અશક્તિ": "weakness", "થાક": "fatigue",
                "ચક્કર": "dizziness", "ખંજવાળ": "itching of skin",
                "સોજો": "skin swelling", "બળતરા": "burning",
                "કબજિયાત": "constipation"
            }
        }
        
        target_map = lang_map.copy()
        if lang_key in colloquial_map:
            target_map.update(colloquial_map[lang_key])

        # Sort keys by length (Longest First) to avoid partial matching greediness
        sorted_items = sorted(target_map.items(), key=lambda x: len(x[0]), reverse=True)
        temp_text = text_lower
        for key, value in sorted_items:
            # We use lower() to be safe, though keys should already be clean
            k_low = key.lower().strip()
            if k_low in temp_text:
                found.add(value)
                # Mark this part of text as "used" so we don't match sub-parts
                temp_text = temp_text.replace(k_low, " " * len(k_low))
                
    return list(found)

@app.route("/api/extract_symptoms", methods=["POST"])
@login_required
def extract_symptoms():
    data = request.json
    text = data.get("text", "")
    lang_code = data.get("language", "en")
    
    print(f"[SYMPTOM EXTRACTION] Input text: '{text}', Language: {lang_code}")
    
    # 1. Local Fast-Path (Keyword Match)
    local_found = local_extract_symptoms(text, lang_code)
    print(f"[LOCAL MATCH] Found: {local_found}")
    
    # 2. AI Extraction (Always try for better accuracy, even if local found something)
    ai_found = []
    try:
        # Enhanced AI prompt with full symptom list
        symptoms_str = ", ".join(symptoms_list)
        
        system_prompt = f"""You are a medical symptom detector. The patient may speak in English, Hindi, or Gujarati.
For Hindi, understand everyday spoken Hindi (Hindustani), not just formal/Sanskritized terms.
For Gujarati, understand everyday spoken Gujarati as used in Saurashtra/Kathiyawad.
        
Your task:
1. Understand the patient's input text
2. Extract ONLY the symptoms mentioned
3. Match them EXACTLY to symptom names from this official list: {symptoms_str}
4. Return a valid JSON array of matched symptom names

Example Input: "मुझे सिर दर्द और बुखार है"
Example Output: ["headache", "fever"]

IMPORTANT: 
- Return ONLY the JSON array, no explanations
- Use exact symptom names from the list
- If no symptoms match, return []
"""
        
        full_prompt = f"{system_prompt}\n\nPatient Input: {text}\n\nExtracted Symptoms (JSON array):"
        
        content, ai_error = call_ai_text(full_prompt)
        if ai_error:
            print(f"[AI ERROR] {ai_error}")
            content = "[]"
        else:
            content = content.strip()
        print(f"[AI RAW RESPONSE] {content}")
        
        # Clean up the response
        if "```" in content:
            content = content.replace("```json", "").replace("```python", "").replace("```", "")
        content = content.strip()
        
        # Try to parse JSON
        extracted = json.loads(content)
        
        # Verify extracted symptoms are actually in our full list
        ai_found = [s for s in extracted if s in symptoms_list]
        print(f"[AI MATCH] Found: {ai_found}")
        
    except Exception as e:
        print(f"[AI ERROR] {e}")
    
    # 3. Combine results (prefer AI, but include local matches too)
    combined = list(set(local_found + ai_found))
    
    if combined:
        source = "ai" if ai_found else "local"
        print(f"[FINAL RESULT] {combined} (source: {source})")
        return jsonify({"status": "success", "symptoms": combined, "source": source})
    else:
        print("[NO MATCH] No symptoms detected")
        return jsonify({"status": "error", "message": "No symptoms detected. Please try speaking more clearly or use manual selection.", "symptoms": []})






@app.route("/api/explain_disease", methods=["POST"])
@login_required
def explain_disease():
    data = request.json
    disease = data.get("disease")
    language = data.get("language", "English") # Already passed as 'English', 'Hindi', or 'Gujarati'
    
    if not disease:
        return jsonify({"error": "No disease specified"})

    try:
        # Provider-agnostic AI call for detailed step-by-step explanation
        register = _language_register_instruction(language)
        prompt = (
            f"Provide a comprehensive, detailed medical explanation for '{disease}' in {language}. "
            f"{register} "
            "Structure your response as follows:\n\n"
            "1. OVERVIEW: What is this condition? (2-3 sentences)\n"
            "2. CAUSES: What causes this disease? List main causes\n"
            "3. SYMPTOMS: Detailed list of common symptoms\n"
            "4. DIAGNOSIS: How is it diagnosed?\n"
            "5. TREATMENT: Step-by-step treatment approach\n"
            "6. PREVENTION: How to prevent it?\n"
            "7. WHEN TO SEE A DOCTOR: Red flags that require immediate medical attention\n"
            "8. LIFESTYLE RECOMMENDATIONS: Diet, exercise, and daily care tips\n\n"
            "Make it VERY detailed and informative. "
            "Aim for approximately 400-500 words in total."
        )
        explanation, ai_error = call_ai_text(prompt)
        if ai_error:
            raise RuntimeError(ai_error)
        return jsonify({"explanation": explanation})

    except Exception as e:
        print(f"AI Error (Explain): {e}")
        # Localized Offline Fallback Response - also detailed
        if language == "Gujarati":
            mock_msg = f"""⚠️ [વિસ્તૃત સમજૂતી - ઓફલાઇન મોડ]

**રોગ:** {disease}

**1. ઝાંખી:**
{disease} એક તબીબી સ્થિતિ છે જે યોગ્ય ધ્યાન અને સારવારની જરૂર છે. આ સ્થિતિ વિવિધ કારણોસર થઈ શકે છે.

**2. કારણો:**
• જીવનશૈલીના પરિબળો
• આનુવંશિક વલણ
• પર્યાવરણીય પરિબળો

**3. લક્ષણો:**
• સામાન્ય અસ્વસ્થતા
• શરીરમાં દુખાવો
• થાક અને નબળાઈ

**4. નિદાન:**
તબીબી પરીક્ષા અને જરૂરી પરીક્ષણો દ્વારા નિદાન થાય છે.

**5. સારવાર:**
પગલું 1: ડોક્ટરની સલાહ લો
પગલું 2: સૂચવેલ દવાઓ લો
પગલું 3: આરામ કરો
પગલું 4: નિયમિત ફોલોઅપ

**6. નિવારણ:**
• સ્વસ્થ આહાર લો
• નિયમિત કસરત કરો
• તણાવ ઘટાડો

**7. ક્યારે ડોક્ટરને મળવું:**
• લક્ષણો વધુ ખરાબ થાય
• તીવ્ર દુખાવો અનુભવો
• તાવ વધે

**8. જીવનશૈલી સલાહ:**
• પુષ્કળ પાણી પીવો
• પૂરતી ઊંઘ લો
• આહારમાં તાજા ફળો અને શાકભાજી

*(AI સેવા હાલમાં મર્યાદિત હોવાથી આ એક પૂર્વનિર્ધારિત સંદેશ છે.)*"""
        elif language == "Hindi":
            mock_msg = f"""⚠️ [विस्तृत स्पष्टीकरण - ऑफलाइन मोड]

**रोग:** {disease}

**1. अवलोकन:**
{disease} एक चिकित्सा स्थिति है जिस पर उचित ध्यान और उपचार की आवश्यकता है। यह स्थिति विभिन्न कारणों से हो सकती है।

**2. कारण:**
• जीवनशैली कारक
• आनुवंशिक प्रवृत्ति
• पर्यावरणीय कारक

**3. लक्षण:**
• सामान्य असुविधा
• शरीर में दर्द  
• थकान और कमजोरी

**4. निदान:**
चिकित्सा परीक्षण और आवश्यक जांच द्वारा निदान किया जाता है।

**5. उपचार:**
चरण 1: डॉक्टर से परामर्श लें
चरण 2: निर्धारित दवाएं लें
चरण 3: आराम करें
चरण 4: नियमित फॉलोअप

**6. रोकथाम:**
• स्वस्थ आहार लें
• नियमित व्यायाम करें
• तनाव कम करें

**7. डॉक्टर से कब मिलें:**
• लक्षण बिगड़ जाएं
• तीव्र दर्द हो
• बुखार बढ़े

**8. जीवनशैली सुझाव:**
• भरपूर पानी पिएं
• पर्याप्त नींद लें
• आहार में ताजे फल और सब्जियां

*(चूंकि AI सेवा वर्तमान में सीमित है, यह एक पूर्व-निर्धारित संदेश है।)*"""
        else:
            mock_msg = f"""⚠️ [DETAILED EXPLANATION - OFFLINE MODE]

**Disease:** {disease}

**1. OVERVIEW:**
{disease} is a medical condition that requires proper attention and treatment. This condition may occur due to various factors and should be addressed promptly.

**2. CAUSES:**
• Lifestyle factors
• Genetic predisposition
• Environmental factors
• Underlying health conditions

**3. SYMPTOMS:**
• General discomfort
• Body aches and pains
• Fatigue and weakness
• Specific organ-related symptoms

**4. DIAGNOSIS:**
Diagnosis is made through medical examination, patient history, and necessary laboratory tests or imaging studies.

**5. TREATMENT:**
Step 1: Consult with a qualified healthcare provider
Step 2: Take prescribed medications as directed
Step 3: Get adequate rest and recovery time
Step 4: Follow up regularly with your doctor
Step 5: Monitor symptoms and report changes

**6. PREVENTION:**
• Maintain a healthy, balanced diet
• Exercise regularly
• Manage stress effectively
• Get adequate sleep
• Avoid risk factors

**7. WHEN TO SEE A DOCTOR:**
• Symptoms worsen suddenly
• Severe pain or discomfort
• High fever persists
• New concerning symptoms appear

**8. LIFESTYLE RECOMMENDATIONS:**
• Drink plenty of water (8-10 glasses daily)
• Include fresh fruits and vegetables in diet
• Maintain regular sleep schedule
• Practice stress-reduction techniques
• Avoid smoking and limit alcohol

*(This is a simulated detailed response as the AI service is currently at capacity. For accurate, personalized medical advice, please consult a healthcare professional.)*"""
        return jsonify({"explanation": mock_msg})

@app.route("/api/generate_report", methods=["POST"])
@login_required
def generate_report():
    data = request.json
    disease = data.get("disease")
    language = data.get("language", "English")
    
    if not disease:
        return jsonify({"error": "No disease specified"})
        
    try:
        doctor_name = session["user"]["username"]
        role = session["user"]["role"]
        
        # 1. Generate text with AI - Explicitly asking for NO MARKDOWN symbols
        prompt = (
            f"Write a professional medical screening report for '{disease}' in {language}. "
            f"Attending: {role} {doctor_name}. "
            "IMPORTANT: Do not use any Markdown symbols like asterisks (**), hashes (##), or pipe tables (|---|). "
            "Use clear headings like 'PHASE 1: FINDINGS', 'PHASE 2: ADVICE', etc. "
            "Keep it clinical and structured with bullet points. Maximum 150 words."
        )
        
        try:
            report_text, ai_error = call_ai_text(prompt)
            if ai_error:
                raise RuntimeError(ai_error)
        except Exception as ai_err:
            print(f"AI Report Error: {ai_err}")
            # Multilingual Fallback
            if language == "Hindi":
                report_text = f"मेडिकल स्क्रीनिंग रिपोर्ट: {disease}\n\nउपस्थित डॉक्टर: {role} {doctor_name}\n\nनिष्कर्ष: रोगी में {disease} के सामान्य लक्षण दिखाई दे रहे हैं।\nसलाह: किसी विशेषज्ञ से मिलें। आराम करें और पर्याप्त पानी पिएं।\nनोट: यह एक सिम्युलेटेड रिपोर्ट है।"
            elif language == "Gujarati":
                report_text = f"મેડિકલ સ્ક્રિનિંગ રિપોર્ટ: {disease}\n\nહાજર ડોક્ટર: {role} {doctor_name}\n\nતારણો: દર્દીમાં {disease} ના સામાન્ય લક્ષણો જોવા મળે છે.\nસલાહ: નિષ્ણાત ડોક્ટરની સલાહ લો. આરામ કરો અને પૂરતું પાણી પીવો.\nનોંધ: આ એક સિમ્યુલેટ રિપોર્ટ છે."
            else:
                report_text = f"MEDICAL SCREENING REPORT: {disease}\n\nAttending: {role} {doctor_name}\n\nFindings: Patient exhibits typical symptoms of {disease}. \nAdvice: Follow up with a specialist. Maintain rest and hydration.\nNote: Simulated report."

        # 2. Generate PDF with Structured Layout
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Select font based on language
        pdf_font = "Helvetica" # Default
        if language in ["Hindi", "Gujarati"] and USING_UNICODE_FONT:
            pdf_font = "Nirmala"
            
        # Custom Title Style
        headers = REPORT_HEADERS.get(language, REPORT_HEADERS["English"])
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName=f"{pdf_font}-Bold" if pdf_font == "Helvetica" else pdf_font,
            fontSize=18,
            alignment=1,
            spaceAfter=20
        )
        
        # Custom Body Style
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['BodyText'],
            fontName=pdf_font,
            fontSize=11,
            leading=16
        )

        # Header Info Style
        header_style = ParagraphStyle(
            'ReportHeader',
            parent=styles['Normal'],
            fontName=pdf_font,
            fontSize=10,
            leading=14
        )
        
        # Initialize story list BEFORE using it
        story = []
        story.append(Paragraph(headers["title"], title_style))
        
        # Header Info Table-like structure
        header_text = (
            f"<b>Disease:</b> {disease}<br/>"
            f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d')}<br/>"
            f"<b>Attending {role}:</b> {doctor_name}<br/>"
            f"<b>Language:</b> {language}<br/>"
        )
        story.append(Paragraph(header_text, header_style))
        story.append(Paragraph("<hr/>", styles["Normal"]))
        
        # 3. Add Structured Information Card
        structured_info = get_detailed_info(disease)
        
        # Format the AI content
        clean_content = report_text.replace("**", "").replace("__", "").replace("###", "").replace("##", "").replace("#", "")
        formatted_content = clean_content.replace('\n', '<br />')
        story.append(Paragraph(formatted_content, body_style))
        
        # Add Sections
        story.append(Spacer(1, 15))
        
        # Medications
        if 'medications' in structured_info and structured_info['medications']:
            story.append(Paragraph(f"<b>{headers['meds']}</b>", body_style))
            meds_text = "<br/>• ".join(structured_info['medications'])
            story.append(Paragraph(f"• {meds_text}", body_style))
            story.append(Spacer(1, 10))

        # Precautions
        if 'precautions' in structured_info and structured_info['precautions']:
            story.append(Paragraph(f"<b>{headers['prec']}</b>", body_style))
            prec_text = "<br/>• ".join(structured_info['precautions'])
            story.append(Paragraph(f"• {prec_text}", body_style))
            story.append(Spacer(1, 10))

        # Diet
        if 'diets' in structured_info and structured_info['diets']:
            story.append(Paragraph(f"<b>{headers['diet']}</b>", body_style))
            diet_text = "<br/>• ".join(structured_info['diets'])
            story.append(Paragraph(f"• {diet_text}", body_style))
            story.append(Spacer(1, 10))

        # Workout
        if 'workouts' in structured_info and structured_info['workouts']:
            story.append(Paragraph(f"<b>{headers['work']}</b>", body_style))
            work_text = "<br/>• ".join(structured_info['workouts'])
            story.append(Paragraph(f"• {work_text}", body_style))
            story.append(Spacer(1, 10))
        
        # Add Disclaimer
        story.append(Spacer(1, 20))
        story.append(Paragraph("<hr/>", styles["Normal"]))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Italic'],
            fontName=pdf_font,
            fontSize=9
        )
        story.append(Paragraph(f"<i>{headers['disclaimer']}</i>", disclaimer_style))
        
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"Medical_Report_{disease}_{language}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"Report Generation Error: {e}")
        return jsonify({"error": str(e)})

@app.route("/api/analyze_image", methods=["POST"])
@login_required
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"})
        
    try:
        prompt = "Analyze this medical image or report and suggest possible condition and advice."
        image_bytes = file.read()
        result, ai_error = call_ai_image(prompt, image_bytes, mime_type=file.mimetype or "image/png")
        if ai_error:
            raise RuntimeError(ai_error)
        return jsonify({"result": result})
        
    except Exception as e:
        print(f"AI Error (Image): {e}")
        # Offline Fallback Response
        mock_result = f"""⚠️ [OFFLINE MODE] AI Analysis Unavailable.

Reason: {e}

        **Simulated Analysis Report:**
        
        **1. Observation:**
        - The uploaded image appears to be a medical scan or report.
        - No critical abnormalities detected in this simulation.
        
        **2. Advice:**
        - Please consult a radiologist or specialist for a real diagnosis.
        - Ensure the image is clear and high-resolution for better analysis when online.
        
        *(This is a placeholder result to demonstrate the UI layout.)*"""
        return jsonify({"result": mock_result})

@app.route("/api/add_patient", methods=["POST"])
@login_required
@role_required("doctor")
def add_patient():
    data = request.json
    # Simple auto-inc ID simulation or just count
    count = patients_col.count_documents({})
    patient_id = f"P{1001 + count}"
    
    patient_data = {
        "patient_id": patient_id,
        "full_name": data.get("full_name"),
        "age": data.get("age"),
        "gender": data.get("gender"),
        "phone_number": data.get("phone_number"),
        "address": data.get("address"),
        "blood_group": data.get("blood_group"),
        "emergency_contact": data.get("emergency_contact")
    }
    patients_col.insert_one(patient_data)
    return jsonify({"status": "success", "patient_id": patient_id})

@app.route("/api/add_history", methods=["POST"])
@login_required
@role_required("doctor")
def add_history():
    data = request.json
    history_entry = {
        "patient_id": data.get("patient_id"),
        "visit_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "symptoms": data.get("symptoms", []),
        "diagnosis": data.get("diagnosis"),
        "test_results": data.get("test_results"),
        "prescribed_medicines": data.get("prescribed_medicines", []),
        "doctor_name": session["user"]["username"],
        "notes": data.get("notes")
    }
    history_col.insert_one(history_entry)
    return jsonify({"status": "success"})

@app.route("/api/user_history", methods=["GET"])
@login_required
def get_user_history():
    if not USING_MONGODB:
        return jsonify({"error": "Database Offline", "history": []})
        
    try:
        # Fetch last 50 predictions for this user
        history = list(predictions_col.find(
            {"username": session["user"]["username"]},
            {"_id": 0}
        ).sort("timestamp", -1).limit(50))
        
        # Format timestamps for JSON
        for h in history:
            if "timestamp" in h:
                h["timestamp"] = h["timestamp"].strftime("%Y-%m-%d %H:%M")
                
        return jsonify({"status": "success", "history": history})
    except Exception as e:
        return jsonify({"error": str(e), "history": []})

@app.route("/api/get_history/<patient_id>", methods=["GET"])
@login_required
def get_history(patient_id):
    records = list(history_col.find({"patient_id": patient_id}, {"_id": 0}))
    patient = patients_col.find_one({"patient_id": patient_id}, {"_id": 0})
    return jsonify({"patient": patient, "history": records})

@app.route("/api/dictionary", methods=["GET"])
def dictionary():
    # Return dictionary data for the table
    # Returning all might be large, but let's try.
    # Structure: [{"english": "fever", "hindi": "...", "gujarati": "..."}]
    
    term_list = []
    
    # Use Hindi keys as the base iteration if available
    hi_trans = TRANSLATIONS.get("hi", {})
    gu_trans = TRANSLATIONS.get("gu", {})
    
    # Create a reverse map for Guj: Value(English) -> Key(Gujarati)
    # This is because the JSON is { "hi": { "hindi_word": "english_word" } }
    # Wait, checking app.py load_translations:
    # json structure is likely { "hi": { "hindi_word": "english_word" }, "gu": { "guj_word": "english_word" } }
    # So to build a row  [English, Hindi, Gujarati], we need to group by English Word.
    
    english_map = {} # english_word -> { hi: ..., gu: ... }
    
    for lang, mapping in TRANSLATIONS.items():
        for local_word, eng_word in mapping.items():
            eng_word = eng_word.strip().lower()
            if eng_word not in english_map:
                english_map[eng_word] = {"en": eng_word, "hi": "-", "gu": "-"}
            
            if lang == "hi":
                english_map[eng_word]["hi"] = local_word
            elif lang == "gu":
                english_map[eng_word]["gu"] = local_word
                
    return jsonify({"terms": list(english_map.values())})

@app.route("/api/find_doctors", methods=["POST"])
@login_required
def find_doctors():
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    disease = data.get("disease")
    
    if not lat or not lng:
        return jsonify({"error": "Missing location data"})
    
    # Optional: Default disease if missing, though unlikely in this flow
    if not disease:
        disease = "General"
        
    result = find_doctors_for_disease(lat, lng, disease)
    return jsonify(result)

@app.route("/api/symptom_map", methods=["GET"])
@login_required
def get_symptom_map():
    # Return mapping of English -> Local for frontend display
    return jsonify(REVERSE_TRANSLATIONS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)    