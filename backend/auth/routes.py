from fastapi import APIRouter, Depends, status
from .models import UserCreate, UserLogin, UserResponse, TokenVerify, GoogleLogin
from .service import AuthService
from .middleware import FirebaseTokenBearer
from sqlalchemy.orm import Session
from . import service, models
from user_profiles.database import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])
token_auth_scheme = FirebaseTokenBearer()

@router.post("/register", response_model=models.UserResponse)
async def register(user_data: models.UserCreate, db: Session = Depends(get_db)):
    return await service.AuthService.register_user(user_data, db)

@router.post("/login")
async def login(user_data: models.UserLogin, db: Session = Depends(get_db)):
    return await service.AuthService.login_user(user_data, db)

@router.post("/google-login")
async def google_login(token_data: models.GoogleLogin, db: Session = Depends(get_db)):
    return await service.AuthService.google_login(token_data, db)

@router.post("/verify-token")
async def verify_token(token_data: TokenVerify):
    """Xác minh token"""
    return await AuthService.verify_token(token_data.token)

@router.get("/me")
async def get_current_user(user_data: dict = Depends(token_auth_scheme)):
    """Lấy thông tin người dùng hiện tại"""
    return {
        "uid": user_data["uid"],
        "email": user_data.get("email", ""),
        "display_name": user_data.get("name", "")
    }