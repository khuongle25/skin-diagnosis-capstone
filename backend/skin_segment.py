import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import tempfile
from utils import preprocess_for_pytorch_segmentation, predict_with_pytorch, mask_to_base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image(image_path, model):
    """
    Process image with segmentation model
    
    Args:
        image_path: Path to input image
        model: Loaded segmentation model
    
    Returns:
        dict: Contains segmented_path and other info
    """
    try:
        # Read image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Preprocess for PyTorch model
        image_tensor, original_img, error = preprocess_for_pytorch_segmentation(image_data)
        
        if error:
            logger.info(f"❌ Preprocessing error: {error}")
            return None
        
        # Get device
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        logger.info(f"SkinSegment: Sử dụng thiết bị {device}")
        
        # Predict mask
        mask_array = predict_with_pytorch(model, device, image_tensor, target_shape=(224, 224))
        
        if mask_array is None:
            logger.info("❌ Segmentation prediction failed")
            return None
        
        # Convert mask to base64
        mask_b64 = mask_to_base64(mask_array)
        
        # Save segmented mask to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            # Convert mask to image and save
            mask_img = (mask_array[0, :, :, 0] * 255).astype(np.uint8)
            cv2.imwrite(tmp_file.name, mask_img)
            seg_tmp_path = tmp_file.name
        
        return {
            'segmented_path': seg_tmp_path,
            'mask_base64': mask_b64,
            'success': True
        }
        
    except Exception as e:
        logger.info(f"❌ Error in process_image: {e}")
        return None

# Thêm hàm để load model PyTorch
def get_pytorch_segmentation_model():
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights=None,  # Không cần pretrained weights
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    return model

def load_pytorch_segmentation_model(model_path):
    device = torch.device("cpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"SkinSegment: Sử dụng thiết bị {device}")
    model = get_pytorch_segmentation_model()
    model = model.to(device)
    
    # Load trọng số đã lưu
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Chuyển sang chế độ đánh giá
    model.eval()
    
    return model, device

def load_models():
    try:    
        pytorch_segmentation_model, device = load_pytorch_segmentation_model('checkpoints/deeplabB7.pth')  
        logger.info("✅ PyTorch segmentation model loaded successfully")  
        return (pytorch_segmentation_model, device)
    except Exception as e:
        logger.info(f"Error loading models: {e}")
        return None
