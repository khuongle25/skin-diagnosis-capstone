# utils.py
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_bytes, img_size=128):
    """Process image for classification model"""
    try:
        # Load image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Ensure image is RGB
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Image must be RGB, got shape: {img_array.shape}")
        
        # Resize image
        img_array = cv2.resize(img_array, (img_size, img_size))
        
        # Normalize
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    
    except Exception as e:
        return None, str(e)

def preprocess_for_segmentation(image_bytes, img_size=256):
    """Process image for segmentation model"""
    try:
        # Load image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize image
        img_array = cv2.resize(img_array, (img_size, img_size))
        
        # Normalize
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    
    except Exception as e:
        return None, str(e)

def mask_to_base64(mask_array):
    """Convert mask array to base64 string"""
    # Scale to 0-255
    mask_img = (mask_array[0, :, :, 0] * 255).astype(np.uint8)
    
    # Create image from array
    mask_pil = Image.fromarray(mask_img)
    
    # Save to bytes
    buffer = BytesIO()
    mask_pil.save(buffer, format="PNG")
    
    # Convert to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def original_to_base64(img_array):
    """Convert normalized image array back to base64 string"""
    # Scale back to 0-255
    img = (img_array[0] * 255).astype(np.uint8)
    
    # Create image from array
    img_pil = Image.fromarray(img)
    
    # Save to bytes
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    
    # Convert to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Add these functions to your utils.py file

def preprocess_for_diagnosis(img_array, mask_array, img_size=224):
    """Process segmented lesion for diagnosis model"""
    try:
        # Get the original image and mask
        img = img_array[0]  # Remove batch dimension
        mask = mask_array[0, :, :, 0]  # Get mask without batch dimension
        
        # Apply mask to focus on the lesion
        masked_img = img.copy()
        for i in range(3):  # Apply to each channel
            masked_img[:, :, i] = masked_img[:, :, i] * mask
        
        # Resize to the diagnosis model's expected size
        masked_img = cv2.resize(masked_img, (img_size, img_size))
        
        # Add batch dimension back
        masked_img = np.expand_dims(masked_img, axis=0)
        
        return masked_img, None
    
    except Exception as e:
        return None, str(e)

def get_lesion_class_mapping():
    """Return mapping of class indices to lesion types"""
    # Replace with your actual class mapping from the diagnosis model
    return {
        0: "Actinic keratosis (akiec)",
        1: "Basal cell carcinoma (bcc)",
        2: "Benign keratosis-like lesions (bkl)",
        3: "Dermatofibroma (df)",
        4: "Melanoma (mel)",
        5: "Melanocytic nevi (nv)",
        6: "Vascular lesions (vasc)"
    }
    
# Thêm vào utils.py
def preprocess_for_diagnosis_with_metadata(img_array, mask_array, metadata=None, img_size=224):
    """Process segmented lesion for diagnosis model with metadata"""
    try:
        # Get the original image and mask
        img = img_array[0]  # Remove batch dimension
        mask = mask_array[0, :, :, 0]  # Get mask without batch dimension
        
        # Apply mask to focus on the lesion
        masked_img = img.copy()
        for i in range(3):  # Apply to each channel
            masked_img[:, :, i] = masked_img[:, :, i] * mask
        
        # Resize to the diagnosis model's expected size
        masked_img = cv2.resize(masked_img, (img_size, img_size))
        
        # Add batch dimension back
        masked_img = np.expand_dims(masked_img, axis=0)
        
        # Nếu không có metadata, trả về chỉ ảnh
        if metadata is None:
            return masked_img, None
            
        # Nếu có metadata, xử lý và trả về cả hai
        processed_metadata = process_user_metadata(metadata)
        return [masked_img, processed_metadata], None
    
    except Exception as e:
        return None, str(e)

def process_user_metadata(metadata):
    """Xử lý metadata từ người dùng thành tensor phù hợp cho mô hình"""
    # Giả sử metadata có dạng:
    # {
    #    "age": 45,
    #    "gender": "male", # hoặc "female", "unknown"
    #    "location": "back" # vị trí tổn thương
    # }
    
    # Khởi tạo mảng zeros có 19 phần tử (hoặc số lượng phù hợp với mô hình)
    metadata_array = np.zeros((1, 19))
    
    # Xử lý giới tính (one-hot encoding)
    if metadata.get("gender") == "female":
        metadata_array[0, 0] = 1  # sex_female
    elif metadata.get("gender") == "male":
        metadata_array[0, 1] = 1  # sex_male
    else:
        metadata_array[0, 2] = 1  # sex_unknown
    
    # Xử lý vị trí tổn thương (giả sử các vị trí từ 3-17)
    location_mapping = {
        "abdomen": 3, "back": 4, "chest": 5, "ear": 6, "face": 7,
        "foot": 8, "genital": 9, "hand": 10, "lower extremity": 11,
        "neck": 12, "scalp": 13, "trunk": 14, "upper extremity": 15,
        "unknown": 16
    }
    
    location = metadata.get("location", "unknown").lower()
    if location in location_mapping:
        metadata_array[0, location_mapping[location]] = 1
    else:
        metadata_array[0, 16] = 1  # unknown location
    
    # Xử lý tuổi (normalized)
    age = metadata.get("age", 50)  # mặc định 50 tuổi nếu không có
    metadata_array[0, 18] = float(age) / 100.0  # chuẩn hóa tuổi
    
    return metadata_array

def preprocess_for_pytorch_segmentation(image_data):
    try:
        # Import tường minh tại đây để đảm bảo numpy luôn khả dụng trong hàm
        import numpy as np
        import cv2
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Đọc ảnh
        if isinstance(image_data, str):
            img = cv2.imread(image_data)
        else:
            # Nếu image_data là dữ liệu bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if img is None:
            return None, None, "Không thể đọc được ảnh"
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Lưu ảnh gốc cho preprocess_for_diagnosis
        original_img = img.copy()
        
        # Định nghĩa tiền xử lý
        preprocess = A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Áp dụng tiền xử lý
        img_processed = preprocess(image=img)["image"]
        return img_processed.unsqueeze(0), original_img, None
    except ImportError as e:
        return None, None, f"Lỗi import thư viện: {str(e)}"
    except Exception as e:
        return None, None, f"Lỗi xử lý ảnh: {str(e)}"

def predict_with_pytorch(model, device, image_tensor, target_shape=(224, 224)):
    """Dự đoán mask và trả về theo định dạng tương thích với model Keras cũ"""
    try:
        # Import tường minh
        import cv2
        import numpy as np
        import torch
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            pred_mask = output.cpu().numpy().squeeze()
        
        # Chuyển thành mask nhị phân với ngưỡng 0.5
        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # Resize mask về kích thước mong muốn (thường là 224x224 cho diagnosis model)
        resized_mask = cv2.resize(binary_mask, target_shape)
        
        # Định dạng kết quả giống với định dạng của model Keras
        # [batch, height, width, channel]
        formatted_mask = np.expand_dims(np.expand_dims(resized_mask, axis=0), axis=-1)
        
        return formatted_mask
    except ImportError as e:
        logger.info(f"Lỗi import thư viện trong predict_with_pytorch: {str(e)}")
        raise e
    except Exception as e:
        logger.info(f"Lỗi trong quá trình dự đoán: {str(e)}")
        raise e