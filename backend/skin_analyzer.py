# skin_analyzer.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
from enum import Enum
import logging
import warnings

from skin_full_seg import SkinSegmentationModel

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinImageType(Enum):
    SKIN_CLOSEUP = "Ảnh da đạt chuẩn"         # Ảnh cận cảnh da, phù hợp cho chẩn đoán
    SKIN_NONSPECIFIC = "Ảnh da không cụ thể"   # Ảnh có vùng da nhưng không phải cận cảnh
    NOT_SKIN = "Không phải ảnh da"            # Không phải ảnh da

class AdvancedSkinClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b5', num_classes=1):
        super(AdvancedSkinClassifier, self).__init__()
        
        # Backbone
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        # Lấy số features
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            in_features = self.backbone.num_features
        
        # Multi-scale feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Color-specific attention
        self.color_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
        # Texture-specific attention
        self.texture_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
        # Color analysis branch
        self.color_branch = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Texture analysis branch
        self.texture_branch = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Surface detection branch
        self.surface_branch = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(160, 128),  # 64 + 64 + 32
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        avg_features = self.global_pool(features).flatten(1) if len(features.shape) == 4 else features
        max_features = self.global_max_pool(features).flatten(1) if len(features.shape) == 4 else features
        
        # Apply attention
        color_attention_weights = self.color_attention(avg_features)
        texture_attention_weights = self.texture_attention(max_features)
        
        color_attended = avg_features * color_attention_weights
        texture_attended = max_features * texture_attention_weights
        
        # Multi-branch analysis
        color_features = self.color_branch(color_attended)
        texture_features = self.texture_branch(texture_attended)
        surface_features = self.surface_branch(avg_features)
        
        # Combine features
        combined = torch.cat([color_features, texture_features, surface_features], dim=1)
        
        # Final prediction
        output = self.classifier(combined)
        
        return output

class SmartSkinDetector:
    def __init__(self):
        self.skin_detector_built = False
        self._setup_skin_ranges()
        
    def _setup_skin_ranges(self):
        """Định nghĩa các range màu da thông minh"""
        
        # RGB ranges cho da (bao gồm cả da sáng và tối)
        self.rgb_ranges = {
            'light_skin': {
                'min': np.array([180, 120, 100]),  # Da sáng
                'max': np.array([255, 200, 180])
            },
            'medium_skin': {
                'min': np.array([120, 80, 60]),    # Da trung bình
                'max': np.array([220, 170, 140])
            },
            'dark_skin': {
                'min': np.array([60, 40, 30]),     # Da tối
                'max': np.array([140, 100, 80])
            },
            'pink_skin': {
                'min': np.array([180, 130, 130]),  # Da hồng (ISIC)
                'max': np.array([255, 210, 200])
            }
        }
        
        # HSV ranges
        self.hsv_ranges = {
            'general_skin': {
                'min': np.array([0, 20, 80]),      # H: 0-25 (yellow-red)
                'max': np.array([25, 255, 255])
            },
            'pink_skin': {
                'min': np.array([300, 15, 100]),   # H: 300-359 (pink-red)
                'max': np.array([359, 80, 255])
            }
        }
        
        # LAB ranges
        self.lab_ranges = {
            'skin_tone': {
                'min': np.array([20, 120, 120]),   # L: brightness, A: green-red, B: blue-yellow
                'max': np.array([200, 150, 150])
            }
        }
        
        # Non-skin materials to reject
        self.non_skin_signatures = {
            'wood': {
                'rgb_mean_range': ([100, 60, 30], [160, 120, 80]),
                'hsv_h_range': (10, 30),
                'texture_uniform': True
            },
            'sky': {
                'rgb_mean_range': ([150, 180, 200], [200, 230, 255]),
                'hsv_h_range': (200, 240),
                'texture_uniform': True
            },
            'fabric': {
                'std_threshold': 15,  # Uniform texture
                'edge_density_low': True
            }
        }
        
        self.skin_detector_built = True
        
    def calculate_skin_probability(self, image_path):
        """Tính xác suất là da bằng smart detector với error handling"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                # Thử với PIL
                try:
                    pil_img = Image.open(image_path).convert('RGB')
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.info(f"Lỗi khi chuyển đổi ảnh: {e}")
                    return 0.0
            
            # Resize cho consistent
            img = cv2.resize(img, (224, 224))
            
            # Convert color spaces
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Lấy center region
            center_mask = self._create_center_mask(img.shape[:2])
            
            # Extract features
            features = self._extract_features(img_rgb, img_hsv, img_lab, center_mask)
            
            # Tính scores
            color_score = self._calculate_color_score(features)
            texture_score = self._calculate_texture_score(img_rgb, center_mask)
            non_skin_penalty = self._calculate_non_skin_penalty(features)
            
            # Combine scores
            final_score = (color_score * 0.6 + texture_score * 0.4) * (1 - non_skin_penalty)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.info(f"Lỗi khi tính xác suất là da: {e}")
            return 0.0
    
    def _create_center_mask(self, shape, center_ratio=0.8):
        """Tạo mask cho vùng trung tâm"""
        h, w = shape
        center_h = int(h * center_ratio)
        center_w = int(w * center_ratio)
        
        start_h = (h - center_h) // 2
        start_w = (w - center_w) // 2
        
        mask = np.zeros((h, w), dtype=bool)
        mask[start_h:start_h+center_h, start_w:start_w+center_w] = True
        
        return mask
    
    def _extract_features(self, img_rgb, img_hsv, img_lab, mask):
        """Extract color features"""
        features = {}
        
        # RGB features
        rgb_pixels = img_rgb[mask]
        features['rgb_mean'] = np.mean(rgb_pixels, axis=0)
        features['rgb_std'] = np.std(rgb_pixels, axis=0)
        features['rgb_pixels'] = rgb_pixels
        
        # HSV features
        hsv_pixels = img_hsv[mask]
        features['hsv_mean'] = np.mean(hsv_pixels, axis=0)
        features['hsv_pixels'] = hsv_pixels
        
        # LAB features
        lab_pixels = img_lab[mask]
        features['lab_mean'] = np.mean(lab_pixels, axis=0)
        features['lab_pixels'] = lab_pixels
        
        return features
    
    def _calculate_color_score(self, features):
        """Tính điểm màu sắc"""
        scores = []
        
        # RGB matching
        rgb_mean = features['rgb_mean']
        rgb_score = 0.0
        
        for skin_type, ranges in self.rgb_ranges.items():
            if np.all(rgb_mean >= ranges['min']) and np.all(rgb_mean <= ranges['max']):
                if skin_type == 'pink_skin':
                    rgb_score = max(rgb_score, 0.9)  # ISIC style
                elif skin_type == 'light_skin':
                    rgb_score = max(rgb_score, 0.8)
                else:
                    rgb_score = max(rgb_score, 0.7)
        
        # Percentage of pixels in skin range
        rgb_pixels = features['rgb_pixels']
        skin_pixel_count = 0
        total_pixels = len(rgb_pixels)
        
        for pixel in rgb_pixels:
            for skin_type, ranges in self.rgb_ranges.items():
                if np.all(pixel >= ranges['min']) and np.all(pixel <= ranges['max']):
                    skin_pixel_count += 1
                    break
        
        pixel_ratio_score = skin_pixel_count / total_pixels if total_pixels > 0 else 0
        scores.append(pixel_ratio_score)
        
        # HSV H channel check (skin tone)
        hsv_pixels = features['hsv_pixels']
        h_values = hsv_pixels[:, 0]
        
        # Skin H values: 0-25 (yellow-red) hoặc 300-359 (pink-red)
        skin_h_count = np.sum((h_values <= 25) | (h_values >= 300))
        h_score = skin_h_count / len(h_values) if len(h_values) > 0 else 0
        scores.append(h_score)
        
        # HSV saturation check (not too gray)
        s_values = hsv_pixels[:, 1]
        good_saturation = np.sum((s_values >= 20) & (s_values <= 200))
        s_score = good_saturation / len(s_values) if len(s_values) > 0 else 0
        scores.append(s_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_texture_score(self, img_rgb, mask):
        """Tính điểm texture"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # Apply mask
            masked_gray = gray.copy()
            masked_gray[~mask] = 0
            
            # Calculate texture features
            
            # 1. Standard deviation (skin has moderate variation)
            std_val = np.std(gray[mask])
            std_score = 1.0 if 10 < std_val < 60 else max(0, 1 - abs(std_val - 35) / 35)
            
            # 2. Edge density (skin has moderate edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges[mask]) / np.sum(mask)
            edge_score = 1.0 if 5 < edge_density < 40 else max(0, 1 - abs(edge_density - 20) / 20)
            
            # 3. Local Binary Pattern-like check
            # Skin has organic, non-uniform patterns
            h, w = gray.shape
            if h > 50 and w > 50:
                center_region = gray[h//4:3*h//4, w//4:3*w//4]
                local_std = np.std(center_region)
                organic_score = min(local_std / 30, 1.0)  # Organic variation
            else:
                organic_score = 0.5
            
            return np.mean([std_score, edge_score, organic_score])
            
        except Exception as e:
            logger.info(f"Lỗi khi tính điểm texture: {e}")
            return 0.5
    
    def _calculate_non_skin_penalty(self, features):
        """Tính penalty cho non-skin materials"""
        penalty = 0.0
        
        rgb_mean = features['rgb_mean']
        rgb_std = features['rgb_std']
        hsv_mean = features['hsv_mean']
        
        # Wood detection
        wood_range = self.non_skin_signatures['wood']['rgb_mean_range']
        if (wood_range[0][0] <= rgb_mean[0] <= wood_range[1][0] and
            wood_range[0][1] <= rgb_mean[1] <= wood_range[1][1] and
            wood_range[0][2] <= rgb_mean[2] <= wood_range[1][2]):
            # Check if H is in wood range
            h_val = hsv_mean[0]
            if 10 <= h_val <= 30:  # Brown/wood hue
                penalty += 0.8
        
        # Sky detection
        sky_range = self.non_skin_signatures['sky']['rgb_mean_range']
        if (sky_range[0][0] <= rgb_mean[0] <= sky_range[1][0] and
            sky_range[0][1] <= rgb_mean[1] <= sky_range[1][1] and
            sky_range[0][2] <= rgb_mean[2] <= sky_range[1][2]):
            h_val = hsv_mean[0]
            if 200 <= h_val <= 240:  # Blue hue
                penalty += 0.9
        
        # Uniform texture (fabric, flat surfaces)
        avg_std = np.mean(rgb_std)
        if avg_std < 15:  # Too uniform
            penalty += 0.3
        
        # Very high saturation (artificial colors)
        s_val = hsv_mean[1]
        if s_val > 200:
            penalty += 0.2
        
        # Very low brightness (too dark)
        v_val = hsv_mean[2]
        if v_val < 50:
            penalty += 0.4
        
        return min(penalty, 0.9)  # Max penalty 90%

class CNNSkinDetector:
    def __init__(self, model_path=None):
        # Cài đặt device
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Thiết lập thư mục lưu backbones
        os.makedirs('checkpoints/backbones', exist_ok=True)
        torch_home = os.environ.get('TORCH_HOME')
        os.environ['TORCH_HOME'] = os.path.abspath('checkpoints/backbones')
        
        # Khởi tạo model
        self.model = AdvancedSkinClassifier('efficientnet_b5', num_classes=1)
        
        # Tải pretrained weights nếu có
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint)
            except Exception as e:
                logger.info(f"Lỗi load model: {e}")
        
        # Move model to device và đặt eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform cho inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Khôi phục TORCH_HOME
        if torch_home:
            os.environ['TORCH_HOME'] = torch_home
    
    def predict_skin_probability(self, image_path):
        """Dự đoán xác suất là da bằng CNN"""
        try:
            # Load và preprocess ảnh
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception:
                return 0.0
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = output.item()
            
            # Cleanup GPU memory
            del image_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return probability
            
        except Exception as e:
            logger.info(f"Lỗi khi dự đoán xác suất là da: {e}")
            # Cleanup nếu có lỗi
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return 0.0

class SkinImageAnalyzer:
    """Phân tích và phân loại ảnh da sử dụng hybrid method"""
    
    def __init__(self, model_path=None, segmentation_model_path="checkpoints/skin_full.pth"):
        """
        Khởi tạo analyzer
        
        Args:
            model_path: Đường dẫn đến model weights (nếu có)
            segmentation_model_path: Đường dẫn đến model segmentation
        """
        # Khởi tạo các thành phần
        self.smart_detector = SmartSkinDetector()
        self.cnn_detector = CNNSkinDetector(model_path)
        
        # Khởi tạo mô hình segmentation
        try:
            self.segmentation_model = SkinSegmentationModel(
                checkpoint_path=segmentation_model_path,
                image_size=512,
                threshold=0.5
            )
            self.segmentation_available = True
            logger.info(f"Đã khởi tạo mô hình segmentation thành công từ {segmentation_model_path}")
        except Exception as e:
            logger.info(f"Lỗi khi khởi tạo mô hình segmentation: {e}")
            self.segmentation_available = False
        
        # Thiết lập thresholds
        self.cnn_threshold = 0.67       # Ngưỡng CNN Score
        self.smart_threshold = 0.4      # Ngưỡng Smart Detector Score
        self.skin_ratio_threshold = 0.2  # Ngưỡng tỷ lệ da trên ảnh (20%)
    
    def analyze_image(self, image_path, verbose=False):
        """
        Phân tích ảnh và phân loại thành 3 loại:
        1. Ảnh da đạt chuẩn (cận cảnh da)
        2. Ảnh da không cụ thể (có da nhưng không phải cận cảnh)
        3. Không phải ảnh da
        
        Args:
            image_path: Đường dẫn tới ảnh hoặc URL
            verbose: In ra kết quả chi tiết
            
        Returns:
            SkinImageType: Loại ảnh da
        """
        # Kiểm tra xem image_path có phải là URL không
        if image_path.startswith(('http://', 'https://')):
            if verbose:
                logger.info("Phát hiện URL, tiến hành tải ảnh...")
            temp_path = self._download_image(image_path)
            if not temp_path:
                return SkinImageType.NOT_SKIN
            image_path = temp_path
            cleanup_temp = True
        else:
            cleanup_temp = False
            
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(image_path):
                if verbose:
                    logger.info(f"File không tồn tại: {image_path}")
                return SkinImageType.NOT_SKIN
            
            # Bước 1: CNN Model
            if verbose:
                logger.info("Bước 1: Kiểm tra bằng CNN Model...")
            cnn_score = self.cnn_detector.predict_skin_probability(image_path)
            
            if verbose:
                logger.info(f"CNN Score: {cnn_score:.3f} (Threshold: {self.cnn_threshold})")
            
            # Kiểm tra CNN Score
            if cnn_score >= self.cnn_threshold:
                # Bước 2: Smart Skin Detector
                if verbose:
                    logger.info("Bước 2: Kiểm tra bằng Smart Skin Detector...")
                smart_score = self.smart_detector.calculate_skin_probability(image_path)
                
                if verbose:
                    logger.info(f"Smart Score: {smart_score:.3f} (Threshold: {self.smart_threshold})")
                
                # Nếu cả hai điều kiện đều đáp ứng
                if smart_score >= self.smart_threshold:
                    if verbose:
                        logger.info("Kết luận: ẢNH DA ĐẠT CHUẨN")
                    result = SkinImageType.SKIN_CLOSEUP
                else:
                    if verbose:
                        logger.info("Smart Detector không đạt, chuyển sang phân tích vùng da...")
            else:
                if verbose:
                    logger.info("CNN Score không đạt, chuyển sang phân tích vùng da...")
            
            # Bước 3: Phân tích vùng da bằng mô hình segmentation
            if 'result' not in locals():  # Nếu chưa có kết quả từ bước 2
                if verbose:
                    logger.info("Bước 3: Phân tích vùng da bằng mô hình segmentation...")
                
                if not self.segmentation_available:
                    if verbose:
                        logger.info("Mô hình segmentation không khả dụng!")
                    result = SkinImageType.NOT_SKIN
                else:
                    try:
                        # Thực hiện segmentation
                        seg_result = self.segmentation_model.predict(image_path)
                        skin_ratio = seg_result['skin_ratio'] / 100.0  # Chuyển từ phần trăm sang tỷ lệ 0-1
                        
                        if verbose:
                            logger.info(f"Tỷ lệ diện tích da: {skin_ratio:.1%}")
                            logger.info(f"Ngưỡng: {self.skin_ratio_threshold:.1%}")
                        
                        # Đưa ra kết luận dựa vào tỷ lệ diện tích da
                        if skin_ratio >= self.skin_ratio_threshold:
                            if verbose:
                                logger.info("Kết luận: ẢNH DA KHÔNG CỤ THỂ")
                            result = SkinImageType.SKIN_NONSPECIFIC
                        else:
                            if verbose:
                                logger.info("Kết luận: KHÔNG PHẢI ẢNH DA")
                            result = SkinImageType.NOT_SKIN
                        
                    except Exception as e:
                        if verbose:
                            logger.info(f"Lỗi khi phân tích vùng da: {str(e)}")
                        result = SkinImageType.NOT_SKIN
            
            logger.info(f"📊 KẾT QUẢ PHÂN TÍCH: {result.name} - {result.value}")
            return result
            
        finally:
            # Dọn dẹp file tạm nếu có
            if cleanup_temp and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except Exception as e:
                    logger.info(f"Lỗi khi xóa file tạm: {e}")
                    pass
                    
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Sử dụng với cú pháp đơn giản
def classify_skin_image(image_path, model_path=None, verbose=False):
    """
    Hàm tiện ích để phân loại ảnh da
    
    Args:
        image_path: Đường dẫn đến ảnh hoặc URL
        model_path: Đường dẫn đến model weights (nếu có)
        verbose: In thông tin chi tiết
        
    Returns:
        SkinImageType: Loại ảnh da (SKIN_CLOSEUP, SKIN_NONSPECIFIC, NOT_SKIN)
    """
    analyzer = SkinImageAnalyzer(model_path)
    return analyzer.analyze_image(image_path, verbose)
