import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best.keras")
    THRESHOLD = os.getenv("THRESHOLD")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
    MODEL_API_URL = os.getenv("MODEL_API_URL", "")
    MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
