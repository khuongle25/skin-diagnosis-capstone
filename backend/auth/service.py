from firebase_admin import auth
from fastapi import HTTPException, status, Depends
import requests
from .models import UserCreate, UserLogin, UserResponse, GoogleLogin
import os
from user_profiles.database import get_db
from user_profiles.crud import get_profile_by_user_id, create_profile
from sqlalchemy.orm import Session

FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY")

class AuthService:
    @staticmethod
    async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
        try:
            user = auth.create_user(
                email=user_data.email,
                password=user_data.password,
                display_name=user_data.display_name or ""
            )
            
            # Tạo profile cho người dùng mới
            profile = get_profile_by_user_id(db, user.uid)
            if not profile:
                create_profile(db, user.uid, user.display_name, user.email)
            
            return UserResponse(
                uid=user.uid,
                email=user.email,
                display_name=user.display_name,
                is_new_user=True
            )
        except auth.EmailAlreadyExistsError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email đã được đăng ký"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Đăng ký thất bại: {str(e)}"
            )
    
    @staticmethod
    async def login_user(user_data: UserLogin, db: Session = Depends(get_db)):
        """Đăng nhập với email/password qua Firebase Auth REST API"""
        try:
            # Sử dụng Firebase Auth REST API vì Firebase Admin SDK không hỗ trợ xác thực email/password
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
            payload = {
                "email": user_data.email,
                "password": user_data.password,
                "returnSecureToken": True
            }
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=data.get("error", {}).get("message", "Đăng nhập thất bại")
                )
            
            # Lấy thông tin người dùng từ Firebase
            user = auth.get_user_by_email(user_data.email)
            
            # Kiểm tra và tạo profile nếu chưa có
            profile = get_profile_by_user_id(db, user.uid)
            if not profile:
                create_profile(db, user.uid, user.display_name, user.email)
            
            return {
                "id_token": data["idToken"],
                "refresh_token": data["refreshToken"],
                "expires_in": data["expiresIn"],
                "user": {
                    "uid": user.uid,
                    "email": user.email,
                    "display_name": user.display_name
                }
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Đăng nhập thất bại: {str(e)}"
            )
    
    @staticmethod
    async def google_login(token_data: GoogleLogin, db: Session = Depends(get_db)):
        """Xử lý đăng nhập bằng Google OAuth token"""
        try:
            # Xác minh ID token từ Google
            decoded_token = auth.verify_id_token(token_data.id_token)
            
            # Lấy email từ token
            email = decoded_token.get("email")
            
            if not email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="Token không chứa email"
                )
                
            try:
                # Kiểm tra xem email đã tồn tại chưa
                user = auth.get_user_by_email(email)
                # Email đã tồn tại, trả về thông tin người dùng
                is_new_user = False
            except auth.UserNotFoundError:
                # Email chưa tồn tại, tạo tài khoản mới
                user_properties = {
                    'email': email,
                    'email_verified': decoded_token.get('email_verified', False),
                    'display_name': decoded_token.get('name', ''),
                    'photo_url': decoded_token.get('picture', ''),
                    'provider_id': 'google.com',
                }
                user = auth.create_user(**user_properties)
                is_new_user = True
            
            # Kiểm tra và tạo profile nếu chưa có
            profile = get_profile_by_user_id(db, user.uid)
            if not profile:
                create_profile(db, user.uid, user.display_name, user.email)
            
            # Tạo custom token cho frontend
            custom_token = auth.create_custom_token(user.uid)
            
            return {
                "token": custom_token.decode('utf-8'),
                "user": {
                    "uid": user.uid,
                    "email": user.email,
                    "display_name": user.display_name
                }
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Xác thực Google thất bại: {str(e)}"
            )
            
    @staticmethod
    async def verify_token(token: str):
        """Xác thực Firebase ID token và trả về thông tin người dùng"""
        try:
            # Xác thực token với Firebase Admin SDK
            decoded_token = auth.verify_id_token(token)
            
            # Trả về thông tin người dùng từ token
            return {
                "uid": decoded_token.get("uid"),
                "email": decoded_token.get("email", ""),
                "name": decoded_token.get("name", ""),
                "picture": decoded_token.get("picture", ""),
                # Những thông tin khác từ decoded_token nếu cần
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token không hợp lệ: {str(e)}"
            )