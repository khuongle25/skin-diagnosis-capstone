import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Union, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinSegmentationModel:
    """Mô hình phân đoạn vùng da từ ảnh"""
    
    def __init__(
        self, 
        checkpoint_path: str = "checkpoints/skin_full.pth", 
        image_size: int = 512,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Khởi tạo mô hình phân đoạn vùng da
        
        Args:
            checkpoint_path: Đường dẫn đến checkpoint model
            image_size: Kích thước ảnh đầu vào cho model
            threshold: Ngưỡng phân đoạn nhị phân
            device: Thiết bị để chạy mô hình ('cuda' hoặc 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.threshold = threshold
        
        # Thiết lập device
        if device is None:
            self.device = torch.device('cpu')
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"SkinSegmentationModel: Sử dụng thiết bị {self.device}")
        else:
            self.device = torch.device(device)
            
        # Khởi tạo mô hình
        self.model = self._load_model()
        self.transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _load_model(self) -> torch.nn.Module:
        """Tải mô hình từ checkpoint"""
        try:
            # Tạo mô hình với kiến trúc giống với lúc huấn luyện
            model = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b7",
                encoder_weights=None,  # Không cần weights pretrained cho inference
                in_channels=3,
                classes=1,
                activation=None
            )
            
            # Tải trọng số
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Không tìm thấy file checkpoint: {self.checkpoint_path}")
                
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Xử lý các định dạng checkpoint khác nhau
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            # check vram
            # run nvidia-smi for result
            result = os.popen("nvidia-smi").read()
            logger.info(f"VRAM: {result}")
            model = model.to(self.device)
            model.eval()  # Đặt sang chế độ đánh giá
            
            logger.info(f"SkinSegmentationModel: Tải mô hình thành công từ {self.checkpoint_path}")
            # check remaining vram
            result = os.popen("nvidia-smi").read()
            logger.info(f"Remaining VRAM: {result}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải mô hình: {str(e)}")
    
    def preprocess_image(self, image: Union[np.ndarray, bytes, str]) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
        """
        Tiền xử lý ảnh cho mô hình
        
        Args:
            image: Ảnh dưới dạng numpy array, bytes, hoặc đường dẫn
            
        Returns:
            Tuple (tensor ảnh đã xử lý, kích thước gốc, ảnh gốc)
        """
        # Xử lý các kiểu dữ liệu đầu vào khác nhau
        if isinstance(image, bytes):
            # Chuyển bytes thành numpy array
            nparr = np.frombuffer(image, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, str):
            # Tải từ đường dẫn
            img_array = cv2.imread(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            # Giả sử là numpy array
            img_array = image
            
            # Xử lý chuyển đổi RGB/BGR nếu cần
            if img_array.shape[2] == 3:
                # Kiểm tra xem có phải BGR (định dạng OpenCV)
                if isinstance(img_array, np.ndarray) and img_array.dtype == np.uint8:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Xử lý ảnh RGBA
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Xử lý ảnh grayscale
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=2)
            
        # Lưu kích thước gốc
        original_h, original_w = img_array.shape[:2]
        
        # Áp dụng biến đổi
        transformed = self.transform(image=img_array)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)
        
        return img_tensor, (original_h, original_w), img_array
    
    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """
        Phân đoạn vùng da từ ảnh đầu vào
        
        Args:
            image: Ảnh đầu vào dưới dạng numpy array, bytes, hoặc đường dẫn
            
        Returns:
            Dict chứa:
                - mask: Mask nhị phân dưới dạng numpy array
                - skin_ratio: Tỷ lệ phần trăm vùng da trong ảnh
                - original_image: Ảnh gốc dưới dạng numpy array
        """
        try:
            # Tiền xử lý
            img_tensor, (original_h, original_w), original_image = self.preprocess_image(image)
            
            # Chạy inference
            prediction = self.model(img_tensor)
            prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()
            
            # Xử lý kết quả
            binary_mask = (prediction > self.threshold).astype(np.float32)
            
            # Resize mask về kích thước gốc
            original_size_mask = cv2.resize(binary_mask, (original_w, original_h))
            
            # Tính tỷ lệ vùng da
            skin_ratio = float(original_size_mask.mean() * 100)
            
            return {
                "mask": original_size_mask,
                "skin_ratio": skin_ratio,
                "original_image": original_image
            }
            
        except Exception as e:
            raise RuntimeError(f"Lỗi khi dự đoán: {str(e)}")
    
    def create_overlay(self, result: Dict[str, Any], alpha: float = 0.5) -> np.ndarray:
        """
        Tạo hình ảnh chồng lớp mask lên ảnh gốc
        
        Args:
            result: Dict kết quả từ hàm predict()
            alpha: Độ trong suốt của overlay (0.0-1.0)
            
        Returns:
            Hình ảnh chồng lớp dưới dạng numpy array
        """
        original_img = result["original_image"]
        mask = result["mask"]
        
        # Tạo mask màu
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[mask > 0.5] = [0, 255, 0]  # Màu xanh lá cho vùng da
        
        # Tạo overlay
        blended = cv2.addWeighted(original_img, 1, colored_mask, alpha, 0)
        return blended