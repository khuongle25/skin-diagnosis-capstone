from pydantic import BaseModel, EmailStr
from typing import Optional, List

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class GoogleLogin(BaseModel):
    id_token: str

class UserResponse(BaseModel):
    uid: str
    email: str
    display_name: Optional[str] = None
    is_new_user: bool = False

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    
class TokenVerify(BaseModel):
    token: str