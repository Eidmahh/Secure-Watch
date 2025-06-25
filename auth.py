import hashlib
import secrets
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pyotp
import qrcode
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from database import UserModel, LogModel
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class SecurityManager:
    
    def __init__(self):
        self.fernet = None
        self._init_encryption()
    def _init_encryption(self):
        try:
            if settings.encryption_key == "your-32-byte-encryption-key-base64-encoded":
                key = Fernet.generate_key()
                print(f"🔑 Generated encryption key: {key.decode()}")
                self.fernet = Fernet(key)
            else:
                self.fernet = Fernet(settings.encryption_key.encode())
        except Exception as e:
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            print(f"⚠️ Using generated encryption key: {key.decode()}")
    
    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def encrypt_data(self, data: str) -> str:
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            print(f"Encryption error: {e}")
            return data  # Fallback to plain text
    
    def decrypt_data(self, encrypted_data: str) -> str:
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            print(f"Decryption error: {e}")
            return encrypted_data  # Fallback to original data
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

class TwoFactorAuth:
    
    @staticmethod
    def generate_secret() -> str:
        return pyotp.random_base32()
    
    @staticmethod
    def generate_qr_code(username: str, secret: str) -> str:
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username,
            issuer_name="Enhanced Security Surveillance System"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Convert to base64 string
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_base64
    
    @staticmethod
    def verify_totp(token: str, secret: str) -> bool:
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        except Exception:
            return False
    
    @staticmethod
    def send_email_otp(email: str, otp: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = settings.smtp_username
            msg['To'] = email
            msg['Subject'] = "Security Alert - One-Time Password"
            
            body = f"""
            Your One-Time Password (OTP) for Enhanced Security Surveillance System:
            
            OTP: {otp}
            
            This code will expire in 5 minutes.
            
            If you did not request this code, please contact your system administrator immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port)
            server.starttls()
            server.login(settings.smtp_username, settings.smtp_password)
            text = msg.as_string()
            server.sendmail(settings.smtp_username, email, text)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Email OTP error: {e}")
            return False

class RoleBasedAccessControl:
    
    ADMIN_PERMISSIONS = [
        "read_users", "create_users", "update_users", "delete_users",
        "read_alerts", "create_alerts", "update_alerts", "delete_alerts",
        "read_analytics", "read_logs", "access_live_stream",
        "manage_system", "view_all_data"
    ]
    
    GUARD_PERMISSIONS = [
        "read_alerts", "update_alerts", "read_own_data", "view_alert_videos"
    ]
    
    @classmethod
    def get_permissions(cls, role: str) -> list:
        if role == "Admin":
            return cls.ADMIN_PERMISSIONS
        elif role == "Security Guard":
            return cls.GUARD_PERMISSIONS
        else:
            return []
    
    @classmethod
    def has_permission(cls, user_role: str, required_permission: str) -> bool:
        permissions = cls.get_permissions(user_role)
        return required_permission in permissions

# Global security manager
security_manager = SecurityManager()
tfa = TwoFactorAuth()
rbac = RoleBasedAccessControl()

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        payload = security_manager.verify_token(token)
        username = payload.get("sub")
        
        user = UserModel.get_user_by_username(username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if user['is_blocked']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is blocked"
            )
        
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def require_admin(current_user: dict = Depends(get_current_user)):
    """Require admin role"""
    if current_user['role'] != 'Admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_permission(permission: str):
    """Require specific permission"""
    async def permission_checker(current_user: dict = Depends(get_current_user)):
        if not rbac.has_permission(current_user['role'], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker

def log_security_event(request: Request, event: str, user_id: int = None, details: dict = None):
    """Log security events"""
    try:
        ip_address = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        LogModel.log_event(
            event=event,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
    except Exception as e:
        print(f"Logging error: {e}")

# Password strength validation
def validate_password_strength(password: str) -> bool:
    """Validate password strength"""
    if len(password) < 8:
        return False
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    return has_upper and has_lower and has_digit and has_special

# Rate limiting (simple implementation)
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, max_requests: int = 10, window_minutes: int = 1) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
        # Check if limit exceeded
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Global rate limiter
rate_limiter = RateLimiter() 