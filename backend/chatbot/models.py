import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
from langchain_community.embeddings import TensorflowHubEmbeddings
import tarfile
import shutil

# Cấu hình logging
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Singleton để quản lý tải model một lần duy nhất"""
    _instance = None
    _models = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Khởi tạo ModelRegistry lần đầu tiên")
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Khởi tạo và tải mô hình"""
        if self._initialized:
            logger.info("ModelRegistry đã được khởi tạo trước đó")
            return
            
        logger.info("Bắt đầu tải mô hình...")
        self._models = {}
        
        # Tải Universal Sentence Encoder
        self._load_universal_sentence_encoder()
        
        self._initialized = True
        logger.info("ModelRegistry đã khởi tạo xong")
    
    def _load_universal_sentence_encoder(self):
        """Tải Universal Sentence Encoder từ file tar.gz đã lưu"""
        try:
            # Đường dẫn đến thư mục models (đồng cấp với thư mục chatbot)
            current_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục chatbot
            backend_dir = os.path.dirname(current_dir)  # thư mục backend
            model_dir = os.path.join(backend_dir, "checkpoints")  # 
            use_dir = os.path.join(model_dir, "universal-sentence-encoder")
            
            # Đường dẫn đến file tar.gz
            tarball_path = os.path.join(model_dir, "universal-sentence-encoder-tensorflow2-multilingual-v2.tar.gz")
            
            # Kiểm tra các file cần thiết cho TensorFlow SavedModel
            required_files = ["saved_model.pb"]
            variables_dir = os.path.join(use_dir, "variables")
            
            model_exists = (os.path.exists(use_dir) and 
                        os.path.isdir(use_dir) and
                        os.path.exists(os.path.join(use_dir, "saved_model.pb")) and
                        os.path.exists(variables_dir) and
                        os.path.isdir(variables_dir))
            
            if model_exists:
                logger.info(f"Sử dụng mô hình USE đã giải nén tại: {use_dir}")
            else:
                # Kiểm tra nếu file tar.gz tồn tại
                if os.path.exists(tarball_path):
                    logger.info(f"Giải nén mô hình từ: {tarball_path}")
                    
                    # Xóa thư mục cũ nếu tồn tại nhưng không đầy đủ
                    if os.path.exists(use_dir):
                        logger.info(f"Xóa thư mục cũ không đầy đủ: {use_dir}")
                        shutil.rmtree(use_dir)
                    
                    # Tạo thư mục đích
                    os.makedirs(use_dir, exist_ok=True)
                    
                    # Giải nén file tar.gz
                    with tarfile.open(tarball_path, "r:gz") as tar:
                        # Giải nén vào thư mục tạm trước
                        temp_dir = os.path.join(model_dir, "temp_extraction")
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        tar.extractall(path=temp_dir)
                        
                        # Tìm thư mục chứa mô hình thực sự
                        extracted_items = os.listdir(temp_dir)
                        model_source_dir = None
                        
                        # Kiểm tra xem có saved_model.pb ở thư mục gốc không
                        if "saved_model.pb" in extracted_items:
                            model_source_dir = temp_dir
                        else:
                            # Tìm trong các thư mục con
                            for item in extracted_items:
                                item_path = os.path.join(temp_dir, item)
                                if os.path.isdir(item_path):
                                    if os.path.exists(os.path.join(item_path, "saved_model.pb")):
                                        model_source_dir = item_path
                                        break
                        
                        if model_source_dir:
                            # Di chuyển tất cả nội dung từ thư mục nguồn đến thư mục đích
                            for item in os.listdir(model_source_dir):
                                source = os.path.join(model_source_dir, item)
                                dest = os.path.join(use_dir, item)
                                if os.path.isdir(source):
                                    shutil.copytree(source, dest, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(source, dest)
                            
                            logger.info(f"Đã giải nén mô hình thành công vào: {use_dir}")
                        else:
                            raise Exception("Không tìm thấy saved_model.pb trong file tar.gz")
                        
                        # Xóa thư mục tạm
                        shutil.rmtree(temp_dir)
                        
                else:
                    logger.warning(f"Không tìm thấy file tar.gz tại: {tarball_path}")
                    raise Exception("Không tìm thấy file tar.gz")
            
            # Kiểm tra lại mô hình sau khi giải nén
            if not (os.path.exists(os.path.join(use_dir, "saved_model.pb")) and 
                    os.path.exists(os.path.join(use_dir, "variables"))):
                raise Exception("Mô hình sau khi giải nén không đầy đủ")
            
            # Tạo embeddings từ mô hình local
            logger.info(f"Tải mô hình USE từ: {use_dir}")
            self._models['use_embeddings'] = TensorflowHubEmbeddings(
                model_url=use_dir
            )
            logger.info("Đã tải mô hình USE thành công!")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải USE: {str(e)}")
            # Fallback: Tải trực tiếp từ TF Hub
            logger.info("Fallback: Tải USE trực tiếp từ TF Hub")
            self._models['use_embeddings'] = TensorflowHubEmbeddings(
                model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )
    
    def get_embeddings(self):
        """Lấy embeddings model đã tải"""
        if not self._initialized:
            self.initialize()
            
        if 'use_embeddings' not in self._models:
            logger.error("USE embeddings chưa được tải!")
            raise ValueError("Universal Sentence Encoder chưa được tải")
            
        return self._models['use_embeddings']

# Khởi tạo singleton khi import
model_registry = ModelRegistry()

def get_embeddings():
    """Hàm helper để lấy embeddings model"""
    if not model_registry._initialized:
        model_registry.initialize()
    return model_registry.get_embeddings()