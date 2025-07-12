"""
Cấu hình tối ưu hóa RAM và performance cho RAG system
"""

import os
import psutil

# Cấu hình giới hạn RAM
MAX_MEMORY_USAGE_PERCENT = 80  # Giới hạn sử dụng RAM (%)
MAX_TEXT_LENGTH = 50000  # Giới hạn độ dài văn bản (50KB)
MAX_BATCH_SIZE = 10  # Số lượng documents xử lý cùng lúc
EMBEDDING_CHUNK_SIZE = 512  # Kích thước chunk cho embeddings

# Cấu hình tối ưu hóa
ENABLE_MEMORY_MONITORING = True
ENABLE_BATCH_PROCESSING = True
ENABLE_LAZY_LOADING = True
ENABLE_MEMORY_CLEANUP = True
ENABLE_FORCE_CLEANUP = True  # Dọn dẹp mạnh mẽ hơn
ENABLE_AUTO_CLEANUP = True   # Tự động dọn dẹp sau mỗi request

# Cấu hình fallback
FALLBACK_EMBEDDING_MODEL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
FALLBACK_CHUNK_SIZE = 256  # Kích thước chunk nhỏ hơn cho fallback

# Cấu hình dọn dẹp
CLEANUP_FREQUENCY = 1  # Dọn dẹp sau mỗi N requests
FORCE_CLEANUP_THRESHOLD = 85  # Ngưỡng để force cleanup (%)
AUTO_CLEANUP_INTERVAL = 5  # Tự động dọn dẹp sau N giây

# Cấu hình monitoring
MEMORY_CHECK_INTERVAL = 5  # Kiểm tra RAM mỗi N operations
LOG_MEMORY_USAGE = True
LOG_CLEANUP_EVENTS = True  # Log các sự kiện dọn dẹp

def get_system_memory_info():
    """Lấy thông tin RAM của hệ thống"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': round(memory.total / (1024**3), 2),
        'available_gb': round(memory.available / (1024**3), 2),
        'used_gb': round(memory.used / (1024**3), 2),
        'percent': memory.percent
    }

def get_optimal_chunk_size():
    """Tính toán kích thước chunk tối ưu dựa trên RAM hiện tại"""
    memory_info = get_system_memory_info()
    
    if memory_info['total_gb'] < 4:
        # RAM < 4GB: sử dụng chunk nhỏ
        return 256
    elif memory_info['total_gb'] < 8:
        # RAM 4-8GB: sử dụng chunk trung bình
        return 512
    else:
        # RAM > 8GB: sử dụng chunk lớn
        return 1024

def get_optimal_batch_size():
    """Tính toán batch size tối ưu dựa trên RAM hiện tại"""
    memory_info = get_system_memory_info()
    
    if memory_info['total_gb'] < 4:
        return 5
    elif memory_info['total_gb'] < 8:
        return 10
    else:
        return 20

def should_use_fallback_model():
    """Kiểm tra xem có nên sử dụng fallback model không"""
    memory_info = get_system_memory_info()
    return memory_info['percent'] > 90 or memory_info['available_gb'] < 1

# Cấu hình động
def get_dynamic_config():
    """Lấy cấu hình động dựa trên tình trạng hệ thống"""
    return {
        'chunk_size': get_optimal_chunk_size(),
        'batch_size': get_optimal_batch_size(),
        'use_fallback': should_use_fallback_model(),
        'memory_info': get_system_memory_info()
    }

def should_force_cleanup():
    """Kiểm tra xem có nên force cleanup không"""
    memory_info = get_system_memory_info()
    return memory_info['percent'] > FORCE_CLEANUP_THRESHOLD

def get_cleanup_strategy():
    """Lấy chiến lược dọn dẹp dựa trên tình trạng RAM"""
    memory_info = get_system_memory_info()
    
    if memory_info['percent'] > 90:
        return "aggressive"  # Dọn dẹp mạnh mẽ
    elif memory_info['percent'] > 80:
        return "moderate"    # Dọn dẹp vừa phải
    else:
        return "light"       # Dọn dẹp nhẹ

def get_cleanup_config():
    """Lấy cấu hình dọn dẹp động"""
    strategy = get_cleanup_strategy()
    
    if strategy == "aggressive":
        return {
            'gc_passes': 5,
            'clear_cache': True,
            'clear_embeddings': True,
            'force_tensorflow_cleanup': True
        }
    elif strategy == "moderate":
        return {
            'gc_passes': 3,
            'clear_cache': True,
            'clear_embeddings': False,
            'force_tensorflow_cleanup': False
        }
    else:
        return {
            'gc_passes': 1,
            'clear_cache': False,
            'clear_embeddings': False,
            'force_tensorflow_cleanup': False
        } 