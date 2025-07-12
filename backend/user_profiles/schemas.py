from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ProfileBase(BaseModel):
    user_id: str
    display_name: str
    email: str

class ProfileCreate(ProfileBase):
    pass

class Profile(ProfileBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class PostBase(BaseModel):
    title: str
    content: str
    visible: Optional[bool] = True

class PostCreate(PostBase):
    image_data: str
    mask_data: str
    patient_metadata: Dict[str, Any]  # Đổi tên từ metadata
    diagnosis: Dict[str, Any]

class Post(PostBase):
    id: int
    user_id: str
    image_data: str
    mask_data: str
    patient_metadata: Dict[str, Any]  
    diagnosis: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None
    star_count: Optional[int] = 0
    is_starred: Optional[bool] = False
    comment_count: Optional[int] = 0  # Thêm trường này
    
    class Config:
        from_attributes = True

class StarBase(BaseModel):
    post_id: int
    user_id: str

class StarCreate(StarBase):
    pass

class Star(StarBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class CommentBase(BaseModel):
    content: str
    
class CommentCreate(CommentBase):
    post_id: int
    user_id: str

class Comment(CommentBase):
    id: int
    post_id: int
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    user_display_name: Optional[str] = None  # Thêm trường này
    
    class Config:
        from_attributes = True
        
class PostWithoutImage(PostBase):
    id: int
    user_id: str
    patient_metadata: Dict[str, Any]
    diagnosis: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None
    star_count: Optional[int] = 0
    is_starred: Optional[bool] = False
    comment_count: Optional[int] = 0  # Thêm trường này
    
    class Config:
        from_attributes = True