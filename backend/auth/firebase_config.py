import os
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo Firebase Admin SDK
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "skin-creds.json")
try:
    cred = credentials.Certificate(cred_path)
    firebase_app = firebase_admin.initialize_app(cred)
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    raise e

def get_auth():
    return auth