import os
import psycopg2
import logging
from psycopg2.extras import RealDictCursor
from typing import Dict, List,  Optional
import json
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kết nối PostgreSQL từ biến môi trường hoặc giá trị mặc định
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASSWORD", "postgres")
DB_PORT = os.environ.get("DB_PORT", "5432")

def get_db_connection():
    """Tạo và trả về kết nối đến database"""
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def create_tables():
    """Tạo các bảng cần thiết trong database"""
    commands = [
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            lesion_type VARCHAR(255) NOT NULL,
            history TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
    ]
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        for command in commands:
            cur.execute(command)
        
        cur.close()
        conn.commit()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
    finally:
        if conn is not None:
            conn.close()

# Thêm debug log
def save_conversation(user_id: str, conversation_id: str, lesion_type: str, 
                  history: str, created_at: Optional[str] = None, 
                  updated_at: Optional[str] = None):
    """Lưu hoặc cập nhật cuộc hội thoại vào database"""
    conn = None
    try:
        logger.info(f"Saving conversation: {conversation_id}, user_id: {user_id}, lesion_type: {lesion_type}")
        logger.info(f"Saving conversation: {conversation_id}, history length: {len(json.loads(history))}")
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Kiểm tra xem cuộc hội thoại đã tồn tại chưa
        cur.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
        exists = cur.fetchone()
        
        if exists:
            # Cập nhật cuộc hội thoại hiện có
            query = """
            UPDATE conversations 
            SET history = %s, updated_at = %s
            WHERE id = %s
            """
            cur.execute(query, (history, updated_at, conversation_id))
            logger.info(f"Conversation {conversation_id} updated")
        else:
            # Tạo cuộc hội thoại mới  
            query = """
            INSERT INTO conversations (id, user_id, lesion_type, history, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cur.execute(query, (conversation_id, user_id, lesion_type, history, created_at, updated_at or created_at))
            logger.info(f"New conversation {conversation_id} created")
        
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.info(f"ERROR saving conversation: {str(e)}")
        logger.error(f"Error saving conversation: {str(e)}")
        return False
    finally:
        if conn is not None:
            conn.close()

def get_conversation_by_id(conversation_id: str) -> Optional[Dict]:
    """Lấy cuộc hội thoại theo ID"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM conversations WHERE id = %s"
        cur.execute(query, (conversation_id,))
        result = cur.fetchone()
        
        cur.close()
        return result
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        return None
    finally:
        if conn is not None:
            conn.close()

def get_conversations_by_user(user_id: str) -> List[Dict]:
    """Lấy danh sách các cuộc hội thoại của người dùng"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT id, lesion_type, created_at, updated_at 
        FROM conversations 
        WHERE user_id = %s
        ORDER BY updated_at DESC
        """
        cur.execute(query, (user_id,))
        results = cur.fetchall()
        
        cur.close()
        return results
    except Exception as e:
        logger.error(f"Error getting user conversations: {str(e)}")
        return []
    finally:
        if conn is not None:
            conn.close()