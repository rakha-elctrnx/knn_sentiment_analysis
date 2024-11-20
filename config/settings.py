from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'sentiment_analysis')