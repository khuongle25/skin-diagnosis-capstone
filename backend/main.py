import os
from dotenv import load_dotenv
load_dotenv()
import cv2
from datetime import datetime
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import json
import logging
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from chatbot.gemini_chat import HybridSkinLesionChatbot  # noqa: F401
from chatbot.db_utils import get_conversation_by_id, save_conversation
from auth.routes import router as auth_router
from chatbot.models import model_registry
from chatbot.sse_server import get_crawl_events
from chatbot.sse_events import add_crawl_event, get_crawl_events
from skin_segment import load_models
from utils import (preprocess_image, mask_to_base64,
                  original_to_base64, 
                  get_lesion_class_mapping, preprocess_for_pytorch_segmentation, predict_with_pytorch)
from chatbot.rag_chain import SkinLesionRAGChain
from user_profiles.database import engine, get_db
from user_profiles.crud import get_profile_by_user_id
import user_profiles.schemas as schemas
from user_profiles import models
from user_profiles.crud import (create_post, get_post_by_id, get_public_posts, get_user_posts, 
                              update_post, delete_post, add_star_direct, remove_star_direct,
                              create_comment, get_post_comments, 
                              delete_comment, check_if_user_starred, get_public_posts_with_starred)
from user_profiles.schemas import PostCreate, Post, CommentCreate, Comment, StarCreate
from utils_ckpt.download_and_extract_checkpoints import download_and_extract_checkpoints
from skin_analyzer import SkinImageAnalyzer, SkinImageType
from skin_lesion_classifier import SkinLesionClassifier, DermModelWithMeta

load_dotenv()


ONEDRIVE_ZIP_URL = "https://storage.googleapis.com/model-khuonglele/checkpoints.zip"
ZIP_SAVE_PATH = "checkpoints.zip"
CHECKPOINTS_DIR = "checkpoints"

download_and_extract_checkpoints(ONEDRIVE_ZIP_URL, ZIP_SAVE_PATH, CHECKPOINTS_DIR)



load_dotenv()


os.environ["OTEL_PYTHON_DISABLED"] = "true"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# check vram
result = os.popen("nvidia-smi").read()
logger.info(f"VRAM: {result}")

# Khởi tạo sớm khi server bắt đầu
logger.info("Khởi tạo ModelRegistry...")
# cuda is available
logger.info(f"Cuda is available: {torch.cuda.is_available()}")
model_registry.initialize()
logger.info("ModelRegistry đã khởi tạo xong")
logger.info("Hello world")

# Khởi tạo bảng database
logger.info("Tạo bảng database")
SkinLesionRAGChain.create_database_tables()
# Tạo tables nếu chưa tồn tại
logger.info("Tạo bảng database metadarta")
models.Base.metadata.create_all(bind=engine)

# Khởi tạo API key cho Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY not found in environment variables. Using default value.")
    api_key = "YOUR_GEMINI_API_KEY_HERE"
    
conversation_histories = {}

class ProfileResponse(BaseModel):
    user_id: str
    display_name: str
    email: str
    
    class Config:
        orm_mode = True
        
class StackingEnsemble(nn.Module):
    def __init__(self, base_models, num_classes=3, meta_dim=0):
        super(StackingEnsemble, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        self.num_classes = num_classes
        meta_learner_input = len(base_models) * num_classes
        if meta_dim > 0:
            self.meta_processor = nn.Sequential(
                nn.Linear(meta_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            meta_learner_input += 32
        else:
            self.meta_processor = None
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_learner_input, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    def forward(self, x, meta=None):
        base_preds = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                if meta is not None and hasattr(model, 'meta_processor') and model.meta_processor is not None:
                    outputs = model(x, meta)
                else:
                    outputs = model(x)
                probs = F.softmax(outputs, dim=1)
                base_preds.append(probs)
        ensemble_features = torch.cat(base_preds, dim=1)
        if meta is not None and self.meta_processor is not None:
            meta_features = self.meta_processor(meta)
            ensemble_features = torch.cat([ensemble_features, meta_features], dim=1)
        output = self.meta_learner(ensemble_features)
        return output
    
# Import SkinImageAnalyzer từ skin_analyzer và SkinLesionClassifier từ skin_lesion_classifier
@app.get("/api/profile/{user_id}", response_model=ProfileResponse)
async def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    profile = get_profile_by_user_id(db, user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

app.include_router(auth_router)
@app.get("/api/crawl-events")
async def crawl_events(request: Request):
    return await get_crawl_events(request)

# Định nghĩa model metadata
class PatientMetadata(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    
# Load models
deeplabv3_model = load_models()

# Khởi tạo SkinImageAnalyzer
# Thay đổi đường dẫn model_path phù hợp với hệ thống của bạn
logger.info("Khởi tạo SkinImageAnalyzer")
skin_analyzer = SkinImageAnalyzer(model_path="checkpoints/is_skin.pth")

# Khởi tạo SkinLesionClassifier
# Thay đổi đường dẫn model_path phù hợp với hệ thống của bạn
MODEL_9CLASS_PATH = "checkpoints/vit_meta_multiclass_tokenmix_best.pth"
MODEL_3CLASS_PATH = "checkpoints/stacking_ensemble_model.pth"
meta_features = ['age_approx'] + [f'site_{i}' for i in range(9)] + [f'sex_{i}' for i in range(3)]
logger.info("Khởi tạo SkinLesionClassifier")

skin_lesion_classifier = SkinLesionClassifier(
    model_9class_path=MODEL_9CLASS_PATH,
    model_3class_path=MODEL_3CLASS_PATH,
    meta_feature_names=meta_features,
    use_seg=True,
    download_backbone=True
)

# Get class mapping
class_mapping = get_lesion_class_mapping()
logger.info("Khởi tạo class mapping")
@app.get("/")
async def root():
    return {"message": "Skin Lesion Analysis API"}

@app.post("/analyze/")
async def classify_image(file: UploadFile = File(...)):
    """Endpoint kiểm tra xem ảnh có phải ảnh da không (nâng cao)"""
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    # Đọc nội dung file
    contents = await file.read()
    
    # Lưu vào file tạm để SkinImageAnalyzer có thể xử lý
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        # Sử dụng SkinImageAnalyzer để phân tích ảnh
        result_type = skin_analyzer.analyze_image(temp_path, verbose=False)
        
        # Xử lý ảnh cho hiển thị
        img_array, error = preprocess_image(contents)
        if error:
            raise HTTPException(status_code=400, detail=f"Error processing image: {error}")
        
        # Convert ảnh thành base64 để hiển thị trên frontend
        img_base64 = original_to_base64(img_array)
        
        # Đưa ra kết quả dựa trên phân loại
        if result_type == SkinImageType.SKIN_CLOSEUP:
            # Ảnh da đạt chuẩn - is_skin=True
            return {
                "is_skin": True,
                "confidence": 0.95,  # Độ tin cậy cao
                "image": img_base64,
                "message": "Ảnh da đạt chuẩn, phù hợp để chẩn đoán"
            }
        elif result_type == SkinImageType.SKIN_NONSPECIFIC:
            # Ảnh có da nhưng không cụ thể - is_skin=False
            return {
                "is_skin": False,
                "confidence": 0.6,  # Độ tin cậy trung bình
                "image": img_base64,
                "message": "Ảnh có vùng da nhưng không phải cận cảnh. Vui lòng chụp lại gần hơn và tập trung vào tổn thương."
            }
        else:  # NOT_SKIN
            # Không phải ảnh da - is_skin=False
            return {
                "is_skin": False,
                "confidence": 0.1,  # Độ tin cậy thấp
                "image": img_base64,
                "message": "Hình ảnh không chứa vùng da. Vui lòng chụp ảnh vùng da cần chẩn đoán."
            }
    
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
    
    finally:
        # Xóa file tạm thời sau khi sử dụng
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error deleting temp file: {e}")
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

@app.post("/classify/")
async def segment_image(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """Endpoint phân đoạn và chẩn đoán tổn thương da"""
    # Check if models are loaded
    if deeplabv3_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly")
    
    # Parse metadata if provided
    patient_metadata = None
    has_metadata = False
    if metadata:
        try:
            patient_metadata = json.loads(metadata)
            has_metadata = True
            logger.info(f"Received metadata: {patient_metadata}")
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            # Nếu parse thất bại, tiếp tục mà không có metadata
            pass
    
    # Đọc file
    contents = await file.read()
    
    # Lưu vào file tạm để xử lý
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name
    
    # Tạo file tạm thứ hai cho mask
    mask_path = None
    try:
        # Preprocess image cho model PyTorch
        pytorch_img_tensor, original_img, error = preprocess_for_pytorch_segmentation(contents)
        
        if error:
            raise HTTPException(status_code=400, detail=f"Error processing image for segmentation: {error}")
        
        # Lấy model PyTorch và device từ tuple
        pytorch_model, device = deeplabv3_model
        
        # Dự đoán mask với model PyTorch - đảm bảo kích thước mask là (224, 224)
        mask = predict_with_pytorch(pytorch_model, device, pytorch_img_tensor, target_shape=(224, 224))

        # Kiểm tra mask có vùng tổn thương không
        if np.sum(mask) < 1e-3:  # hoặc np.count_nonzero(mask) == 0
            # DỪNG LUÔN, KHÔNG PHÂN LOẠI, KHÔNG VÀO except Exception ở ngoài
            raise HTTPException(
                status_code=400,
                detail="Không có tổn thương nào được mô hình phân đoạn tìm thấy"
            )

        # Lưu mask vào file tạm để sử dụng với SkinLesionClassifier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_mask_file:
            # Chuyển đổi mask thành hình ảnh và lưu
            mask_img = (mask[0, :, :, 0] * 255).astype(np.uint8)
            cv2.imwrite(temp_mask_file.name, mask_img)
            mask_path = temp_mask_file.name
            logger.info(f"Saved mask to temporary file: {mask_path}")
        
        # QUAN TRỌNG: Luôn tạo metadata_dict với giá trị mặc định
        # Thay vì có thể là None, luôn có các giá trị mặc định
        metadata_dict = {
            'age': 50,  # Mặc định 50 tuổi
            'anatomical_site': 'unknown',  # Mặc định unknown
            'sex': 'unknown'  # Mặc định unknown
        }
        
        # Nếu có metadata từ người dùng, cập nhật metadata_dict
        if has_metadata:
            # Chỉ cập nhật khi có giá trị không rỗng
            if patient_metadata.get('age') is not None:
                metadata_dict['age'] = patient_metadata.get('age')
            if patient_metadata.get('location'):
                metadata_dict['anatomical_site'] = patient_metadata.get('location')
            if patient_metadata.get('gender'):
                metadata_dict['sex'] = patient_metadata.get('gender')
            
            logger.info(f"Processed metadata: {metadata_dict}")
        else:
            logger.info(f"Using default metadata: {metadata_dict}")
        
        try:
            # Phân loại với SkinLesionClassifier - luôn xử lý với metadata (mặc định hoặc người dùng cung cấp)
            classification_result = skin_lesion_classifier.classify(
                image_path=temp_path,
                mask_path=mask_path, 
                metadata=metadata_dict  # Luôn có giá trị, không bao giờ None
            )
            
            # Chuẩn bị đầu ra
            predicted_class = classification_result['predicted_class']
            confidence = classification_result['confidence']
            probabilities = classification_result['probabilities']
            
            # Xử lý top 3 predictions
            top_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            top_predictions = []
            
            for cls, prob in top_classes:
                class_id = skin_lesion_classifier.class_names.index(cls)
                top_predictions.append({
                    "label": cls,
                    "confidence": float(prob),
                    "class_id": class_id
                })
            
            # Chuyển đổi ảnh và mask thành base64 để hiển thị trên frontend
            seg_img_array = np.array([cv2.resize(original_img, (224, 224))])
            seg_img_array = seg_img_array / 255.0
            img_base64 = original_to_base64(seg_img_array)
            mask_base64 = mask_to_base64(mask)
            
            return {
                "is_skin": True,
                "confidence": 0.95,
                "image": img_base64,
                "mask": mask_base64,
                "diagnosis": {
                    "label": predicted_class,
                    "confidence": confidence,
                    "class_id": skin_lesion_classifier.class_names.index(predicted_class),
                    "top_predictions": top_predictions
                },
                "metadata_used": has_metadata
            }
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise
        
    except HTTPException as e:
        # Nếu là HTTPException (ví dụ lỗi 400 ở trên), raise lại luôn để FastAPI trả về đúng mã lỗi
        raise e
    except Exception as e:
        logger.error(f"Error in segment_image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Xóa file tạm
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if mask_path and os.path.exists(mask_path):
                os.unlink(mask_path)
        except Exception as e:
            logger.error(f"Error deleting temp files: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
               
# API CHO CHATBOT, KHÔNG LIÊN QUAN PHÂN LOẠI ẢNH!!

# Tạo model cho request chat
class ChatRequest(BaseModel):
    message: str
    lesion_type: str
    user_id: str
    conversation_id: Optional[str] = None
    enhance_knowledge: Optional[bool] = False
    
class ChatHistoryRequest(BaseModel):
    user_id: str
    
@app.post("/api/initialize-chatbot")
async def initialize_chatbot(request: dict):
    try:
        lesion_type = request.get("lesion_type")
        user_id = request.get("user_id") or "anonymous"
        if not lesion_type:
            return JSONResponse(status_code=400, content={"error": "lesion_type is required"})
        conversation_id = str(uuid.uuid4())
        # Lưu conversation rỗng vào DB
        save_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            lesion_type=lesion_type,
            history=json.dumps([]),
            created_at=datetime.now().isoformat()
        )
        return {
            "conversation_id": conversation_id,
            "lesion_type": lesion_type,
            "success": True,
            "message": "Chatbot initialized successfully"
        }
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    try:
        user_id = chat_request.user_id
        message = chat_request.message
        lesion_type = chat_request.lesion_type
        conversation_id = chat_request.conversation_id
        enhance_knowledge = chat_request.enhance_knowledge

        if not message:
            return JSONResponse(status_code=400, content={"error": "Message is required"})

        # Kiểm tra conversation_id hợp lệ
        if conversation_id:
            conv_data = get_conversation_by_id(conversation_id)
            if not conv_data:
                logger.warning(f"Conversation {conversation_id} not found, will create new conversation")
                conversation_id = str(uuid.uuid4())
                save_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    lesion_type=lesion_type,
                    history=json.dumps([]),
                    created_at=datetime.now().isoformat()
                )
        else:
            conversation_id = str(uuid.uuid4())
            save_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                lesion_type=lesion_type,
                history=json.dumps([]),
                created_at=datetime.now().isoformat()
            )

        conversation = get_conversation_by_id(conversation_id)
        history = json.loads(conversation.get('history', '[]')) if conversation else []

        chatbot = HybridSkinLesionChatbot(api_key, user_id, conversation_id)
        response, ft_answer, rag_answer = chatbot.chat(
            question_vi=message,
            lesion_type=lesion_type,
            enhance_knowledge=enhance_knowledge,
            history=history  # truyền vào đây
        )

        # Lưu vào DB như cũ
        conversation = get_conversation_by_id(conversation_id)
        history = json.loads(conversation.get('history', '[]')) if conversation else []
        history.append({"role": "human", "content": message})
        history.append({"role": "ai", "content": response})
        save_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            lesion_type=lesion_type,
            history=json.dumps(history),
            updated_at=datetime.now().isoformat()
        )

        return {
            "response": response,
            "conversation_id": conversation_id,
            "lesion_type": lesion_type,
            "ft_answer": ft_answer,
            "rag_answer": rag_answer
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/conversations/{user_id}")
async def get_user_conversations(user_id: str):
    """Lấy danh sách các cuộc hội thoại của người dùng"""
    try:
        conversations = SkinLesionRAGChain.get_user_conversations(user_id)
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
# Thêm endpoint mới
@app.get("/api/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Lấy lịch sử trò chuyện theo ID"""
    try:
        conversation = get_conversation_by_id(conversation_id)
        if not conversation:
            return JSONResponse(
                status_code=404,
                content={"error": "Conversation not found"}
            )
        
        # Parse history từ JSON string thành list
        history = json.loads(conversation.get('history', '[]'))
        
        return {
            "conversation_id": conversation_id,
            "lesion_type": conversation.get('lesion_type'),
            "history": history,
            "created_at": conversation.get('created_at'),
            "updated_at": conversation.get('updated_at')
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
@app.get("/api/crawl-events")
async def crawl_events(request: Request):
    return await get_crawl_events(request)

# API endpoints cho Community Feed

@app.post("/api/posts/", response_model=Post)
async def create_new_post(post: PostCreate, db: Session = Depends(get_db)):
    """Tạo bài đăng mới"""
    return create_post(db, post, post.user_id)

@app.get("/api/posts/")
async def read_public_posts(skip: int = 0, limit: int = 20, user_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Lấy danh sách bài đăng công khai (Feed)"""
    if user_id:
        posts = get_public_posts_with_starred(db, user_id, skip=skip, limit=limit)
    else:
        posts = get_public_posts(db, skip=skip, limit=limit)
    return posts

@app.get("/api/posts/user/{user_id}", response_model=list[Post])  # Thay đổi từ PostWithoutImage sang Post
async def read_user_posts(user_id: str, skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    """Lấy danh sách bài đăng của một người dùng"""
    posts = get_user_posts(db, user_id=user_id, skip=skip, limit=limit)
    return posts

@app.get("/api/posts/{post_id}", response_model=Post)
async def read_post(post_id: int, db: Session = Depends(get_db)):
    """Lấy chi tiết một bài đăng theo ID"""
    db_post = get_post_by_id(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    return db_post

@app.put("/api/posts/{post_id}", response_model=Post)
async def update_existing_post(post_id: int, post: PostCreate, db: Session = Depends(get_db)):
    """Cập nhật bài đăng"""
    db_post = get_post_by_id(db, post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    if db_post.user_id != post.user_id:
        raise HTTPException(status_code=403, detail="Không có quyền sửa bài đăng này")
    
    return update_post(db, post_id, post)

@app.delete("/api/posts/{post_id}")
async def delete_existing_post(post_id: int, user_id: str, db: Session = Depends(get_db)):
    """Xóa một bài đăng"""
    # Gọi sửa hàm delete_post với user_id để xác thực người dùng
    success = delete_post(db, post_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài đăng hoặc không có quyền xóa")
    return {"message": "Đã xóa bài đăng thành công"}

@app.put("/api/posts/{post_id}/visibility")
async def update_post_visibility(post_id: int, update_data: dict, db: Session = Depends(get_db)):
    """Cập nhật trạng thái hiển thị của bài đăng"""
    post = get_post_by_id(db, post_id=post_id, detach=True)
    if post is None:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    
    # Thực hiện cập nhật visibility bằng SQL thuần túy để tránh lỗi dict
    db.execute(
        text("UPDATE posts SET visible = :visible WHERE id = :post_id"),
        {"visible": update_data.get("visible"), "post_id": post_id}
    )
    db.commit()
    
    # Lấy lại post đã cập nhật
    updated_post = get_post_by_id(db, post_id=post_id)
    return updated_post

# Thêm class PydanticModel này
class DiagnosisToPost(BaseModel):
    user_id: str
    user_display_name: Optional[str] = None  # Thêm trường này
    title: str
    content: str
    image_data: str
    mask_data: str
    patient_metadata: Dict[str, Any]
    diagnosis: Dict[str, Any]
    visible: bool = True

@app.post("/api/create-post-from-diagnosis/", response_model=Post)
async def create_post_from_diagnosis(post_data: DiagnosisToPost, db: Session = Depends(get_db)):
    """Tạo bài đăng từ kết quả chẩn đoán"""
    # Kiểm tra user_id có tồn tại không
    profile = get_profile_by_user_id(db, post_data.user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin người dùng")
    
    # Kiểm tra dữ liệu hình ảnh
    if not post_data.image_data or not post_data.mask_data:
        raise HTTPException(status_code=400, detail="Thiếu dữ liệu hình ảnh hoặc mask")
    
    # Debug log để kiểm tra metadata
    logger.info(f"Received patient_metadata: {post_data.patient_metadata}")
    
    # Tạo post
    post = create_post(db, 
        schemas.PostCreate(
            title=post_data.title,
            content=post_data.content,
            image_data=post_data.image_data,
            mask_data=post_data.mask_data,
            patient_metadata=post_data.patient_metadata,
            diagnosis=post_data.diagnosis,
            visible=post_data.visible
        ), 
        post_data.user_id
    )
    
    # Nếu có user_display_name từ frontend thì cập nhật
    if post_data.user_display_name:
        setattr(post, 'user_display_name', post_data.user_display_name)
    else:
        # Nếu không lấy từ profile
        setattr(post, 'user_display_name', profile.display_name)
    
    return post

# API endpoints cho Stars (likes)

@app.post("/api/stars/")
async def add_star(star: StarCreate, db: Session = Depends(get_db)):
    """Thêm star cho bài đăng"""
    # Kiểm tra post tồn tại bằng SQL thuần túy thay vì load post
    post_exists = db.query(models.Post.id).filter(models.Post.id == star.post_id).first()
    if not post_exists:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    
    # Sử dụng hàm add_star_direct không qua ORM
    result = add_star_direct(db, star.post_id, star.user_id)
    
    return {
        "id": 0,
        "post_id": star.post_id,
        "user_id": star.user_id,
        "created_at": datetime.now()
    }

@app.delete("/api/stars/{post_id}")
async def remove_star(post_id: int, user_id: str, db: Session = Depends(get_db)):
    """Xóa star (unlike)"""
    # Kiểm tra post tồn tại bằng SQL thuần túy
    post_exists = db.query(models.Post.id).filter(models.Post.id == post_id).first()
    if not post_exists:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    
    # Sử dụng hàm remove_star_direct không qua ORM
    success = remove_star_direct(db, post_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy star để xóa")
    return {"message": "Đã xóa star thành công"}

@app.get("/api/posts/{post_id}/starred", response_model=dict)
async def check_post_starred(post_id: int, user_id: str, db: Session = Depends(get_db)):
    """Kiểm tra xem user đã star bài đăng chưa"""
    post = get_post_by_id(db, post_id=post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    
    is_starred = check_if_user_starred(db, post_id, user_id)
    return {"is_starred": is_starred}

# API endpoints cho Comments

@app.post("/api/comments/", response_model=Comment)
async def add_comment(comment: CommentCreate, db: Session = Depends(get_db)):
    """Thêm comment cho bài đăng"""
    post = get_post_by_id(db, post_id=comment.post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    
    # Kiểm tra profile của user
    profile = get_profile_by_user_id(db, comment.user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin người dùng")
    
    # Tạo comment mới chỉ với user_id
    db_comment = create_comment(db, comment)
    
    # Gắn thêm display_name vào response
    setattr(db_comment, 'user_display_name', profile.display_name)
    
    return db_comment

@app.get("/api/posts/{post_id}/comments", response_model=list[Comment])
async def read_post_comments(post_id: int, db: Session = Depends(get_db)):
    """Lấy danh sách comments của một bài đăng"""
    post = get_post_by_id(db, post_id=post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Bài đăng không tồn tại")
    
    return get_post_comments(db, post_id)

@app.delete("/api/comments/{comment_id}")
async def remove_comment(comment_id: int, user_id: str, db: Session = Depends(get_db)):
    """Xóa comment"""
    success = delete_comment(db, comment_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy comment hoặc không có quyền xóa")
    return {"message": "Đã xóa comment thành công"}
        



if __name__ == "__main__":
    logger.info("Chạy server")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
