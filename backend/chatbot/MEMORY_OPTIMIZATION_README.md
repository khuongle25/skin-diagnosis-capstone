# Tối ưu hóa RAM cho RAG System

## Tổng quan

Hệ thống RAG đã được tối ưu hóa để giảm thiểu việc sử dụng RAM khi xử lý văn bản lớn với Universal Sentence Encoder multilingual model.

## Các tối ưu hóa chính

### 1. Lazy Loading cho Embeddings

- Embeddings model chỉ được tải khi cần thiết
- Tránh tải model ngay khi khởi tạo RAG chain
- Tiết kiệm RAM ban đầu

### 2. Batch Processing

- Xử lý documents theo batch thay vì tất cả cùng lúc
- Kích thước batch được tính toán động dựa trên RAM hệ thống
- Dọn dẹp bộ nhớ giữa các batch

### 3. Text Truncation

- Giới hạn độ dài văn bản (mặc định 50KB)
- Cắt ngắn văn bản quá dài để tránh tràn RAM
- Thêm dấu hiệu "[truncated]" cho văn bản bị cắt

### 4. Memory Monitoring

- Theo dõi sử dụng RAM liên tục
- Tự động dọn dẹp khi RAM vượt quá 80%
- Log thông tin sử dụng RAM

### 5. Dynamic Configuration

- Kích thước chunk được tính toán dựa trên RAM hệ thống
- Batch size tự động điều chỉnh
- Fallback model khi RAM thấp

### 6. Advanced Memory Cleanup ⭐ **MỚI**

- Dọn dẹp RAM triệt để sau mỗi request
- Chiến lược dọn dẹp động (light/moderate/aggressive)
- Context manager tự động dọn dẹp
- Force cleanup khi RAM cao
- Clear TensorFlow session và cache

## Cấu hình

### File `memory_config.py`

```python
# Giới hạn RAM
MAX_MEMORY_USAGE_PERCENT = 80  # %

# Giới hạn văn bản
MAX_TEXT_LENGTH = 50000  # characters

# Kích thước batch mặc định
MAX_BATCH_SIZE = 10

# Bật/tắt các tính năng
ENABLE_MEMORY_MONITORING = True
ENABLE_BATCH_PROCESSING = True
ENABLE_LAZY_LOADING = True
ENABLE_MEMORY_CLEANUP = True
ENABLE_FORCE_CLEANUP = True      # ⭐ MỚI
ENABLE_AUTO_CLEANUP = True       # ⭐ MỚI

# Cấu hình dọn dẹp ⭐ MỚI
CLEANUP_FREQUENCY = 1            # Dọn dẹp sau mỗi N requests
FORCE_CLEANUP_THRESHOLD = 85     # Ngưỡng để force cleanup (%)
AUTO_CLEANUP_INTERVAL = 5        # Tự động dọn dẹp sau N giây
```

### Cấu hình động

Hệ thống tự động điều chỉnh dựa trên RAM:

- **RAM < 4GB**: Chunk size 256, Batch size 5
- **RAM 4-8GB**: Chunk size 512, Batch size 10
- **RAM > 8GB**: Chunk size 1024, Batch size 20

### Chiến lược dọn dẹp ⭐ MỚI

- **Light**: 1 GC pass, không clear cache
- **Moderate**: 3 GC passes, clear cache
- **Aggressive**: 5 GC passes, clear cache + TensorFlow session

## Sử dụng

### Khởi tạo RAG Chain

```python
from rag_chain import SkinLesionRAGChain

# Khởi tạo với lazy loading
rag_chain = SkinLesionRAGChain(
    api_key="your_api_key",
    user_id="user_id",
    conversation_id="conversation_id"
)

# Embeddings sẽ chỉ được tải khi cần
```

### Sử dụng Context Manager (Khuyến nghị) ⭐ MỚI

```python
from rag_chain import rag_chain_context

# Tự động dọn dẹp khi thoát
with rag_chain_context(
    api_key="your_api_key",
    user_id="user_id",
    conversation_id="conversation_id"
) as rag_chain:

    # Văn bản sẽ tự động được tối ưu hóa
    success = rag_chain.initialize_knowledge("MEL", num_search_results=3)
    response = rag_chain.get_response("Câu hỏi", "MEL", enhance_knowledge=True)

# Tự động dọn dẹp khi thoát context
```

### Dọn dẹp thủ công

```python
from rag_chain import cleanup_memory, force_cleanup_memory

# Dọn dẹp cơ bản
cleanup_memory()

# Dọn dẹp mạnh mẽ (khi RAM cao)
force_cleanup_memory()

# Dọn dẹp RAG chain
rag_chain.cleanup()
```

## Monitoring

### Kiểm tra sử dụng RAM

```python
from memory_config import get_system_memory_info, get_cleanup_strategy

memory_info = get_system_memory_info()
strategy = get_cleanup_strategy()

print(f"RAM: {memory_info['percent']}% used")
print(f"Strategy: {strategy}")
```

### Log memory usage

```python
from rag_chain import get_memory_usage, check_memory_limit

# Kiểm tra RAM
if check_memory_limit():
    print("Memory usage high!")

# Lấy thông tin chi tiết
memory = get_memory_usage()
print(f"Memory: {memory['percent']}% used")
```

## Testing

### Test tối ưu hóa cơ bản

```bash
cd backend/chatbot
python test_memory_optimization.py
```

### Test dọn dẹp RAM ⭐ MỚI

```bash
cd backend/chatbot
python test_memory_cleanup.py
```

Script sẽ test:

1. Dọn dẹp RAM cơ bản
2. Dọn dẹp với RAG chain
3. Áp lực RAM cao
4. Chiến lược dọn dẹp
5. Theo dõi theo thời gian

## Troubleshooting

### Vấn đề thường gặp

1. **RAM vẫn cao sau khi chat**:

   - Sử dụng context manager
   - Gọi `force_cleanup_memory()` sau mỗi request
   - Kiểm tra `get_cleanup_strategy()`

2. **RAM vẫn cao**:

   - Giảm `MAX_BATCH_SIZE`
   - Giảm `MAX_TEXT_LENGTH`
   - Tăng `MAX_MEMORY_USAGE_PERCENT`

3. **Performance chậm**:

   - Tăng `MAX_BATCH_SIZE` (nếu RAM cho phép)
   - Tăng chunk size
   - Tắt `ENABLE_MEMORY_MONITORING`

4. **Model không tải**:
   - Kiểm tra API key
   - Kiểm tra kết nối internet
   - Sử dụng fallback model

### Debug

Bật debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Cho hệ thống RAM thấp (< 4GB)**:

   - Sử dụng fallback model
   - Giảm batch size
   - Tăng frequency cleanup
   - Sử dụng context manager

2. **Cho hệ thống RAM cao (> 8GB)**:

   - Tăng batch size
   - Tăng chunk size
   - Giảm frequency cleanup

3. **Production environment**:
   - Monitor memory usage
   - Set up alerts
   - Regular cleanup schedules
   - Sử dụng context manager

## Best Practices ⭐ MỚI

### 1. Luôn sử dụng Context Manager

```python
# ✅ Tốt
with rag_chain_context(...) as rag_chain:
    response = rag_chain.get_response(...)

# ❌ Không tốt
rag_chain = SkinLesionRAGChain(...)
response = rag_chain.get_response(...)
# Quên cleanup
```

### 2. Dọn dẹp định kỳ

```python
# Sau mỗi N requests
if request_count % CLEANUP_FREQUENCY == 0:
    cleanup_memory()
```

### 3. Force cleanup khi cần

```python
# Khi RAM cao
if should_force_cleanup():
    force_cleanup_memory()
```

### 4. Monitor chiến lược dọn dẹp

```python
strategy = get_cleanup_strategy()
if strategy == "aggressive":
    logger.warning("High memory pressure detected")
```

## Metrics

Theo dõi các metrics sau:

- Memory usage percentage
- Text processing time
- Batch processing time
- Embedding loading time
- Cleanup frequency
- Cleanup strategy distribution ⭐ MỚI
- Memory recovery rate ⭐ MỚI

## Changelog

### v1.1.0 ⭐ MỚI

- Thêm advanced memory cleanup
- Thêm context manager
- Thêm chiến lược dọn dẹp động
- Thêm force cleanup
- Thêm TensorFlow session cleanup
- Thêm test memory cleanup

### v1.0.0

- Thêm lazy loading cho embeddings
- Thêm batch processing
- Thêm memory monitoring
- Thêm text truncation
- Thêm dynamic configuration
