from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
import json
from . import models, schemas
from datetime import datetime
from sqlalchemy.orm import make_transient

# Hàm chuyển đổi JSON thành dict cho post
def _process_post(post, detach=False):
    if post is None:
        return None
    
    # Tạo bản sao detached nếu cần
    if detach:
        session = Session.object_session(post)
        if session:
            session.expunge(post)
            make_transient(post)
    
    # Chuyển đổi chuỗi JSON thành dict
    if hasattr(post, 'patient_metadata') and post.patient_metadata:
        try:
            post.patient_metadata = json.loads(post.patient_metadata)
        except:
            post.patient_metadata = {}
            
    if hasattr(post, 'diagnosis') and post.diagnosis:
        try:
            post.diagnosis = json.loads(post.diagnosis)
        except:
            post.diagnosis = {}
    
    return post

# Profile functions
def get_profile_by_user_id(db: Session, user_id: str):
    """Lấy profile theo user_id (Firebase UID)"""
    return db.query(models.Profile).filter(models.Profile.user_id == user_id).first()

def create_profile(db: Session, user_id: str, display_name: str, email: str):
    """Tạo profile mới"""
    db_profile = models.Profile(
        user_id=user_id,
        display_name=display_name,
        email=email
    )
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

# Post functions
def create_post(db: Session, post: schemas.PostCreate, user_id: str):
    """Tạo post mới sau khi chẩn đoán"""
    db_post = models.Post(
        user_id=user_id,
        title=post.title,
        content=post.content,
        image_data=post.image_data,
        mask_data=post.mask_data,
        patient_metadata=json.dumps(post.patient_metadata),  # Chuyển dict thành JSON string để lưu
        diagnosis=json.dumps(post.diagnosis),  # Chuyển dict thành JSON string để lưu
        visible=post.visible,
        created_at=datetime.now()
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return _process_post(db_post)

def get_post_by_id(db: Session, post_id: int, detach=True):
    """Lấy post theo ID"""
    post = db.query(models.Post).filter(models.Post.id == post_id).first()
    return _process_post(post, detach=detach)

def get_public_posts(db: Session, skip: int = 0, limit: int = 20):
    """Lấy danh sách bài đăng công khai"""
    posts = (db.query(models.Post)
              .filter(models.Post.visible == True)
              .order_by(desc(models.Post.created_at))
              .offset(skip)
              .limit(limit)
              .all())
    
    processed_posts = []
    for post in posts:
        processed_post = _process_post(post)
        
        # Đếm số lượng comments cho mỗi post
        comment_count = db.query(func.count(models.Comment.id)).filter(models.Comment.post_id == post.id).scalar()
        setattr(processed_post, 'comment_count', comment_count)
        
        # Lấy tên người dùng tạo post
        user_profile = db.query(models.Profile).filter(models.Profile.user_id == post.user_id).first()
        if user_profile:
            setattr(processed_post, 'user_display_name', user_profile.display_name)
            
        processed_posts.append(processed_post)
    
    return processed_posts

def get_user_posts(db: Session, user_id: str, skip: int = 0, limit: int = 100):
    """Lấy danh sách bài đăng của một người dùng"""
    posts = (db.query(models.Post)
              .filter(models.Post.user_id == user_id)
              .order_by(desc(models.Post.created_at))
              .offset(skip)
              .limit(limit)
              .all())
    
    processed_posts = []
    for post in posts:
        processed_post = _process_post(post, detach=True)
        
        # Giữ lại image_data và mask_data
        processed_post.image_data = post.image_data
        processed_post.mask_data = post.mask_data
        
        # Lấy profile của người đăng bài
        profile = get_profile_by_user_id(db, post.user_id)
        if profile:
            setattr(processed_post, 'user_display_name', profile.display_name)
        
        # QUAN TRỌNG: Đếm số comment chính xác - xác minh query
        comment_count = db.query(func.count(models.Comment.id)).filter(models.Comment.post_id == post.id).scalar()
        setattr(processed_post, 'comment_count', comment_count)
        print(f"Post ID {post.id}: comment count = {comment_count}")
        
        # Đếm stars
        star_count = db.query(func.count(models.Star.id)).filter(models.Star.post_id == post.id).scalar()
        setattr(processed_post, 'star_count', star_count)
        
        # Đảm bảo is_starred luôn được đặt và có kiểu dữ liệu đúng
        is_starred = db.query(models.Star).filter(
            models.Star.post_id == post.id, 
            models.Star.user_id == user_id
        ).first() is not None
        
        # Đảm bảo trường is_starred luôn được set và có dạng JSON boolean
        processed_post.is_starred = is_starred  # Không sử dụng setattr để tránh lỗi
        
        print(f"Post ID {post.id}: is_starred = {is_starred}")
        processed_posts.append(processed_post)
    
    return processed_posts

def update_post(db: Session, post_id: int, post_update: schemas.PostCreate):
    """Cập nhật thông tin bài đăng"""
    db_post = get_post_by_id(db, post_id)
    if db_post:
        db_post.title = post_update.title
        db_post.content = post_update.content
        db_post.visible = post_update.visible
        db_post.updated_at = datetime.now()
        db.commit()
        db.refresh(db_post)
    return _process_post(db_post)

# Cập nhật backend/user_profiles/crud.py - hàm delete_post
def delete_post(db: Session, post_id: int, user_id: str = None):
    """Xóa một bài đăng theo ID"""
    # Tìm post trong database
    query = db.query(models.Post).filter(models.Post.id == post_id)
    
    # Nếu có user_id, thêm điều kiện vào truy vấn để đảm bảo chỉ người dùng tạo post mới có thể xóa
    if user_id:
        query = query.filter(models.Post.user_id == user_id)
    
    db_post = query.first()
    
    if db_post:
        try:
            # Xóa tất cả stars liên quan
            db.query(models.Star).filter(models.Star.post_id == post_id).delete()
            
            # Xóa tất cả comments liên quan
            db.query(models.Comment).filter(models.Comment.post_id == post_id).delete()
            
            # Xóa post
            db.delete(db_post)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Error deleting post: {e}")
            return False
    return False

def create_star(db: Session, star: schemas.StarCreate):
    """Thêm star cho một bài đăng"""
    # Kiểm tra xem user đã star bài này chưa
    existing = (db.query(models.Star)
                 .filter(models.Star.post_id == star.post_id, 
                         models.Star.user_id == star.user_id)
                 .first())
    
    if existing:
        return existing
    
    try:
        # Tạo star mới
        db_star = models.Star(
            post_id=star.post_id,
            user_id=star.user_id
        )
        db.add(db_star)
        db.commit()
        db.refresh(db_star)
        
        # Đếm số lượng star và cập nhật TRỰC TIẾP không qua ORM
        star_count = db.query(func.count(models.Star.id)).filter(models.Star.post_id == star.post_id).scalar()
        
        # Cập nhật CHỈ trường star_count bằng native SQL
        db.execute(
        text("UPDATE posts SET star_count = :count WHERE id = :post_id"),
        {"count": star_count, "post_id": star.post_id}
        )
        
        return db_star
    except Exception as e:
        db.rollback()
        raise e

def delete_star(db: Session, post_id: int, user_id: str):
    """Xóa star (unlike)"""
    db_star = (db.query(models.Star)
                .filter(models.Star.post_id == post_id, 
                        models.Star.user_id == user_id)
                .first())
    if db_star:
        try:
            # Xóa star
            db.delete(db_star)
            db.commit()
            
            # Đếm số lượng star
            star_count = db.query(func.count(models.Star.id)).filter(models.Star.post_id == post_id).scalar()
            
            # Cập nhật CHỈ trường star_count bằng native SQL
            db.execute(
                text("UPDATE posts SET star_count = :count WHERE id = :post_id"),
                {"count": star_count, "post_id": post_id}
            )
            db.commit()
            
            return True
        except Exception as e:
            db.rollback()
            raise e
    return False

def get_post_stars(db: Session, post_id: int):
    """Lấy danh sách stars của một bài đăng"""
    return db.query(models.Star).filter(models.Star.post_id == post_id).all()

def check_if_user_starred(db: Session, post_id: int, user_id: str):
    """Kiểm tra xem user đã star bài đăng chưa"""
    return db.query(models.Star).filter(models.Star.post_id == post_id, models.Star.user_id == user_id).first() is not None

# Comment functions
def create_comment(db: Session, comment: schemas.CommentCreate):
    """Tạo bình luận mới"""
    db_comment = models.Comment(
        post_id=comment.post_id,
        user_id=comment.user_id,
        content=comment.content
        # Loại bỏ user_display_name=comment.user_display_name vì không có trong model
    )
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    return db_comment

def get_post_comments(db: Session, post_id: int):
    """Lấy danh sách comments của một bài đăng và thông tin người dùng"""
    # Sử dụng join để lấy thông tin từ bảng profiles
    comments_with_profiles = (
        db.query(models.Comment, models.Profile.display_name)
        .outerjoin(models.Profile, models.Comment.user_id == models.Profile.user_id)
        .filter(models.Comment.post_id == post_id)
        .order_by(models.Comment.created_at)
        .all()
    )
    
    result = []
    for comment, display_name in comments_with_profiles:
        comment_dict = comment.__dict__.copy()
        if '_sa_instance_state' in comment_dict:
            del comment_dict['_sa_instance_state']
        
        # Thêm display_name vào kết quả
        comment_dict['user_display_name'] = display_name or "Người dùng ẩn danh"
        
        result.append(comment_dict)
    
    return result

def delete_comment(db: Session, comment_id: int, user_id: str):
    """Xóa comment (chỉ user tạo comment mới được xóa)"""
    db_comment = (db.query(models.Comment)
                    .filter(models.Comment.id == comment_id, 
                            models.Comment.user_id == user_id)
                    .first())
    if db_comment:
        db.delete(db_comment)
        db.commit()
        return True
    return False

def get_public_posts_with_starred(db: Session, user_id: str, skip: int = 0, limit: int = 20):
    """Lấy danh sách bài đăng công khai và kiểm tra người dùng đã star chưa"""
    posts = (db.query(models.Post)
              .filter(models.Post.visible == True)
              .order_by(desc(models.Post.created_at))
              .offset(skip)
              .limit(limit)
              .all())
    
    # Xử lý các trường JSON
    processed_posts = [_process_post(post) for post in posts]
    
    # Thêm thuộc tính is_starred và comment_count cho mỗi post
    for post in processed_posts:
        # Kiểm tra star cho mỗi post
        is_starred = check_if_user_starred(db, post.id, user_id)
        setattr(post, 'is_starred', is_starred)
        
        # Đếm số lượng comments cho mỗi post
        comment_count = db.query(func.count(models.Comment.id)).filter(models.Comment.post_id == post.id).scalar()
        setattr(post, 'comment_count', comment_count)
        
        # Lấy tên người dùng tạo post
        user_profile = db.query(models.Profile).filter(models.Profile.user_id == post.user_id).first()
        if user_profile:
            setattr(post, 'user_display_name', user_profile.display_name)
    
    return processed_posts

def add_star_direct(db: Session, post_id: int, user_id: str):
    """Thêm star cho post (phương pháp trực tiếp không qua ORM)"""
    # Kiểm tra đã star chưa
    existing = db.query(models.Star).filter(
        models.Star.post_id == post_id,
        models.Star.user_id == user_id
    ).first()
    
    if existing:
        return existing
        
    try:
        # Insert star trực tiếp bằng SQL - thêm text() để wrap câu lệnh SQL
        db.execute(
            text("INSERT INTO stars (post_id, user_id, created_at) VALUES (:post_id, :user_id, now())"),
            {"post_id": post_id, "user_id": user_id}
        )
        
        # Cập nhật star count - thêm text() để wrap câu lệnh SQL
        db.execute(
            text("UPDATE posts SET star_count = (SELECT COUNT(*) FROM stars WHERE post_id = :post_id) WHERE id = :post_id"),
            {"post_id": post_id}
        )
        
        db.commit()
        
        # Tạo object để trả về
        star = models.Star(
            post_id=post_id,
            user_id=user_id,
            created_at=datetime.now()
        )
        return star
        
    except Exception as e:
        db.rollback()
        raise e

# Cập nhật hàm remove_star_direct
def remove_star_direct(db: Session, post_id: int, user_id: str):
    """Xóa star khỏi post (phương pháp trực tiếp không qua ORM)"""
    try:
        # Xóa star trực tiếp bằng SQL - thêm text() để wrap câu lệnh SQL
        result = db.execute(
            text("DELETE FROM stars WHERE post_id = :post_id AND user_id = :user_id"),
            {"post_id": post_id, "user_id": user_id}
        )
        
        if result.rowcount == 0:
            # Không tìm thấy star để xóa
            return False
            
        # Cập nhật star count - thêm text() để wrap câu lệnh SQL
        db.execute(
            text("UPDATE posts SET star_count = (SELECT COUNT(*) FROM stars WHERE post_id = :post_id) WHERE id = :post_id"),
            {"post_id": post_id}
        )
        
        db.commit()
        return True
        
    except Exception as e:
        db.rollback()
        raise e