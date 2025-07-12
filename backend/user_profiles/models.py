from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, UniqueConstraint
from sqlalchemy.sql import func
from .database import Base

class Profile(Base):
    __tablename__ = "profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    display_name = Column(String)
    email = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("profiles.user_id"), index=True)
    title = Column(String)
    content = Column(Text)
    image_data = Column(Text)  # Base64 encoded image
    mask_data = Column(Text)   # Base64 encoded mask
    patient_metadata = Column(Text)    # JSON string of metadata - Đổi tên từ metadata
    diagnosis = Column(Text)   # JSON string of diagnosis result
    visible = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    star_count = Column(Integer, default=0)

class Star(Base):
    __tablename__ = "stars"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True)
    user_id = Column(String, ForeignKey("profiles.user_id"), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Composite unique constraint to prevent duplicate stars
    __table_args__ = (UniqueConstraint('post_id', 'user_id', name='_post_user_star_uc'),)

class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True)
    user_id = Column(String, ForeignKey("profiles.user_id"), index=True)
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
 