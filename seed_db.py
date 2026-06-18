import os
from pymongo import MongoClient
import datetime
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB Atlas with timeout
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["healthcare_db"]

def seed_data():
    # 1. Clear existing data (Optional, for clean start)
    db.users.delete_many({})
    db.patients.delete_many({})
    db.medical_history.delete_many({})
    
    # 2. Add Sample User (Doctor)
    hashed_pw = generate_password_hash("password123")
    db.users.insert_one({
        "username": "dr_ravi",
        "email": "ravi@example.com",
        "password": hashed_pw,
        "role": "doctor",
        "created_at": datetime.datetime.now()
    })
    print("User 'dr_ravi' created (Password: password123)")

    # 3. Add Sample Patient
    patient_id = "P1001"
    db.patients.insert_one({
        "patient_id": patient_id,
        "full_name": "John Doe",
        "age": 30,
        "gender": "Male",
        "phone_number": "9876543210",
        "address": "123 Street, Ahmedabad",
        "blood_group": "A+",
        "emergency_contact": "9998887776"
    })
    print(f"Patient '{patient_id}' created.")

    # 4. Add Sample Medical History
    db.medical_history.insert_one({
        "patient_id": patient_id,
        "visit_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "symptoms": ["Fever", "Fatigue"],
        "diagnosis": "Common Flu",
        "test_results": "Blood Test - Normal",
        "prescribed_medicines": ["Paracetamol", "Vitamin C"],
        "doctor_name": "Dr. Ravi",
        "notes": "Patient advised 3 days of rest and plenty of fluids."
    })
    print(f"Medical history added for '{patient_id}'.")

if __name__ == "__main__":
    try:
        seed_data()
        print("\n✅ Sample data populated successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
