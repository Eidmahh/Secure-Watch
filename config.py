import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database Configuration
    db_host: str = "localhost"
    db_user: str = "root"
    db_password: str = "eiodfanA1$@$"
    db_name: str = "surveillance_system"
    db_port: int = 3306
    
    # Security Configuration
    secret_key: str = "your-super-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Encryption Key (32 bytes)
    encryption_key: str = "your-32-byte-encryption-key-base64-encoded"
    
    # Email Configuration for 2FA
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = "your-email@gmail.com"
    smtp_password: str = "your-app-password"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    
    # Application Settings
    debug: bool = True
    cors_origins: List[str] = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://192.168.1.26:3000",  # Your local IP for mobile access
        "http://192.168.1.26:8000",  # Backend access
        "*"  # Allow all origins for development (remove in production)
    ]
    
    # Model Paths
    violence_model_path: str = "Models/violence_detector_final.keras"
    pose_model_path: str = "Models/pose_landmarker.task"
    
    class Config:
        env_file = ".env"

settings = Settings() 