# skin_lesion_classifier.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import timm
import joblib
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DermModelWithMeta(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=9, use_seg=True, 
                use_meta=True, meta_dim=0, pretrained=True):
        super(DermModelWithMeta, self).__init__()
        self.use_seg = use_seg
        self.use_meta = use_meta
        in_channels = 3 + (1 if use_seg else 0)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            in_chans=in_channels
        )
        if 'vit' in model_name or 'deit' in model_name or 'swin' in model_name:
            backbone_out = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif 'efficientnet' in model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError(f"Model {model_name} chưa được hỗ trợ")
        self.meta_dim = meta_dim
        if use_meta and meta_dim > 0:
            self.meta_processor = nn.Sequential(
                nn.Linear(meta_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(backbone_out + 64, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            self.meta_processor = None
            self.classifier = nn.Linear(backbone_out, num_classes)
            
    def forward(self, x, meta=None):
        img_features = self.backbone(x)
        if self.use_meta and meta is not None and self.meta_processor is not None:
            meta_features = self.meta_processor(meta)
            combined_features = torch.cat([img_features, meta_features], dim=1)
            output = self.classifier(combined_features)
        else:
            output = self.classifier(img_features)
        return output

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

class SkinLesionClassifier:
    def __init__(self, 
                 model_9class_path, 
                 model_3class_path, 
                 model_9class_arch='vit_base_patch16_224', 
                 model_3class_arch='vit_base_patch16_224',
                 meta_feature_names=None,
                 use_seg=True,
                 download_backbone=True,
                 device=None,
                 meta_classifier_path="checkpoints/meta_classifier.joblib"):
        """
        Khởi tạo classifier với 2 mô hình
        
        Args:
            model_9class_path: Đường dẫn đến checkpoint của mô hình 9 lớp
            model_3class_path: Đường dẫn đến checkpoint của mô hình 3 lớp (hoặc ensemble)
            model_9class_arch: Kiến trúc của mô hình 9 lớp
            model_3class_arch: Kiến trúc của mô hình 3 lớp
            meta_feature_names: List tên các trường metadata sử dụng
            use_seg: Sử dụng segmentation hay không
            download_backbone: Tải và lưu backbone từ timm về thư mục cục bộ
            device: Thiết bị tính toán (None để tự động chọn CUDA nếu có)
        """
        if device is None:
            self.device = torch.device('cpu')
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"SkinLesionClassifier: Sử dụng thiết bị {self.device}")
        else:
            self.device = device
            
        self.model_9class_arch = model_9class_arch
        self.model_3class_arch = model_3class_arch
        self.use_seg = use_seg
        
        # Tên các lớp
        self.class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']
        self.bkl_idx = self.class_names.index('BKL')
        self.nv_idx = self.class_names.index('NV')
        self.mel_idx = self.class_names.index('MEL')
        
        # Lưu tên các trường metadata nếu có
        self.meta_feature_names = meta_feature_names
        self.meta_dim = 0 if meta_feature_names is None else len(meta_feature_names)
        
        # Tạo thư mục lưu trữ backbone nếu cần
        if download_backbone:
            os.makedirs('checkpoints/backbones', exist_ok=True)
            os.environ['TORCH_HOME'] = os.path.abspath('checkpoints/backbones')
        
        # Tải mô hình
        self.model_9class = self.load_model(
            model_9class_path, 
            model_9class_arch, 
            num_classes=9, 
            meta_dim=self.meta_dim
        )
        
        self.model_3class = self.load_model(
            model_3class_path, 
            model_3class_arch, 
            num_classes=3, 
            meta_dim=self.meta_dim
        )
        
        self.meta_classifier = None
        if meta_classifier_path is not None and os.path.exists(meta_classifier_path):
            self.meta_classifier = joblib.load(meta_classifier_path)
            print(f"✓ Đã nạp meta-classifier từ {meta_classifier_path}")
            
        # Transforms cho ảnh đầu vào
        self.transforms = self.get_transforms(use_seg=use_seg)
    
    def load_model(self, model_path, model_architecture, num_classes, meta_dim=0):
        """Tải mô hình từ checkpoint"""
        # check vram
        logger.info(f"Đang tải mô hình từ {model_path}...")
        result = os.popen("nvidia-smi").read()
        logger.info(f"VRAM: {result}")
        try:
            # raise Exception("test")
            if os.path.exists(model_path):
                model = torch.load(model_path, map_location=self.device, weights_only=False)
                
                if isinstance(model, nn.Module):
                    logger.info("✓ Đã tải full model thành công")
                    return model.to(self.device)
                    
                if isinstance(model, dict):
                    logger.info(f"✓ Đã tải checkpoint thành công, đang khởi tạo mô hình...")
                    new_model = DermModelWithMeta(
                        model_name=model_architecture,
                        num_classes=num_classes,
                        use_seg=self.use_seg,
                        use_meta=meta_dim > 0,
                        meta_dim=meta_dim,
                        pretrained=False
                    )
                    
                    if 'model_state_dict' in model:
                        new_model.load_state_dict(model['model_state_dict'])
                        logger.info("✓ Đã tải state_dict từ checkpoint")
                    elif 'state_dict' in model:
                        new_model.load_state_dict(model['state_dict'])
                        logger.info("✓ Đã tải state_dict từ checkpoint")
                    else:
                        try:
                            new_model.load_state_dict(model)
                            logger.info("✓ Đã tải state_dict trực tiếp")
                        except Exception as e:
                            logger.info(f"⚠️ Không tìm thấy state_dict trong checkpoint: {e}")
                            
                    return new_model.to(self.device)
                    
            logger.info("⚠️ File không tồn tại hoặc định dạng không hỗ trợ, đang tạo mô hình mới...")
            new_model = DermModelWithMeta(
                model_name=model_architecture,
                num_classes=num_classes,
                use_seg=self.use_seg,
                use_meta=meta_dim > 0,
                meta_dim=meta_dim,
                pretrained=True
            )
            return new_model.to(self.device)
            
        except Exception as e:
            logger.info(f"❌ Lỗi khi tải model: {e}")
            # check remaining vram
            result = os.popen("nvidia-smi").read()
            logger.info(f"Remaining VRAM: {result}")
            logger.info("Tạo mô hình mới với pretrained weights...")
            new_model = DermModelWithMeta(
                model_name=model_architecture,
                num_classes=num_classes,
                use_seg=self.use_seg,
                use_meta=meta_dim > 0,
                meta_dim=meta_dim,
                pretrained=True
            )
            return new_model.to(self.device)
    
    def get_transforms(self, use_seg=True):
        """Khởi tạo transforms cho ảnh đầu vào"""
        height, width = 224, 224
        n_channels = 3 + (1 if use_seg else 0)
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=[0.485, 0.456, 0.406] + [0.5] * (n_channels - 3),
                      std=[0.229, 0.224, 0.225] + [0.5] * (n_channels - 3)),
            ToTensorV2(),
        ])
    
    def preprocess_image(self, image_path, mask_path=None):
        """Tiền xử lý ảnh"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.use_seg and mask_path is not None and os.path.exists(mask_path):
            seg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if seg_mask.shape[:2] != img.shape[:2]:
                seg_mask = cv2.resize(seg_mask, (img.shape[1], img.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            if seg_mask.max() > 1:
                seg_mask = seg_mask / 255.0
            img = np.dstack((img, seg_mask))
        
        # Apply transforms
        img_tensor = self.transforms(image=img)['image']
        return img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def preprocess_metadata(self, metadata_dict):
        """Tiền xử lý metadata"""
        if self.meta_feature_names is None or metadata_dict is None:
            return None
        
        meta_values = []
        
        for feature in self.meta_feature_names:
            if feature in metadata_dict:
                meta_values.append(float(metadata_dict[feature]))
            else:
                meta_values.append(0.0)  # Default value if feature is missing
                
        meta_tensor = torch.tensor([meta_values], dtype=torch.float32).to(self.device)
        return meta_tensor
    
    def process_age(self, age):
        """Chuyển đổi tuổi thành giá trị chuẩn hóa
        Lưu ý: Cần điều chỉnh min_age và max_age phù hợp với bộ dữ liệu của bạn
        """
        if age is None:
            return 0.5  # Default middle value
        
        min_age = 0  # Adjust as needed based on your dataset
        max_age = 100  # Adjust as needed based on your dataset
        
        return (float(age) - min_age) / (max_age - min_age)
    
    def process_anatomical_site(self, site):
        """Tạo one-hot encoding cho vị trí giải phẫu"""
        sites = ['anterior torso', 'head/neck', 'lateral torso', 
                 'lower extremity', 'oral/genital', 'palms/soles', 
                 'posterior torso', 'upper extremity', 'unknown']
        
        result = [0] * len(sites)
        if site in sites:
            result[sites.index(site)] = 1
        else:
            result[sites.index('unknown')] = 1
            
        return result
    
    def process_sex(self, sex):
        """Tạo one-hot encoding cho giới tính"""
        sexes = ['female', 'male', 'unknown']
        
        result = [0] * len(sexes)
        if sex in sexes:
            result[sexes.index(sex)] = 1
        else:
            result[sexes.index('unknown')] = 1
            
        return result
    
    def process_patient_metadata(self, age=None, anatomical_site=None, sex=None):
        """Xử lý metadata của bệnh nhân"""
        meta_dict = {}
        
        # Xử lý tuổi
        meta_dict['age_approx'] = self.process_age(age)
        
        # Xử lý vị trí giải phẫu
        site_features = self.process_anatomical_site(anatomical_site)
        for i, val in enumerate(site_features):
            meta_dict[f'site_{i}'] = val
            
        # Xử lý giới tính
        sex_features = self.process_sex(sex)
        for i, val in enumerate(sex_features):
            meta_dict[f'sex_{i}'] = val
            
        return meta_dict
    
    def is_bkl_mel_nv_prediction(self, probs, threshold=0.2):
        """Kiểm tra xem dự đoán có thuộc về 3 lớp BKL, MEL, NV hay không"""
        top1_idx = np.argmax(probs)
        top2_idx = np.argsort(probs)[::-1][:2]

        # Nếu xác xuất của top dưới ngưỡng 0.6, đồng thời trị tuyệt đối hiệu xác suất của top 1 và top 2 dưới 0.2, đồng thời top 2 phải thuộc 3 lớp đặc biệt:
        if probs[top1_idx] < 0.6 and abs(probs[top1_idx] - probs[top2_idx[1]]) < 0.4:
            if top2_idx[1] in [self.bkl_idx, self.mel_idx, self.nv_idx]:
                return True
        # Nếu lớp dự đoán chính thuộc 1 trong 3 lớp
        if top1_idx in [self.bkl_idx, self.mel_idx, self.nv_idx]:
            # Lấy top 4 xác suất cao nhất
            top4_idx = np.argsort(probs)[::-1][:4]
            top3_idx = np.argsort(probs)[::-1][:3]
            
            # Kiểm tra xem cả 3 lớp đặc biệt có thuộc top 4 không
            if all(idx in top4_idx for idx in [self.bkl_idx, self.mel_idx, self.nv_idx]):
                return True
            if all(idx in top3_idx for idx in [self.mel_idx, self.nv_idx]):
                return True
            
        return False
    
    def classify(self, image_path, mask_path=None, metadata=None, alpha=0.2, threshold=0.2):
        """Phân loại ảnh tổn thương da
        
        Args:
            image_path: Đường dẫn đến ảnh
            mask_path: Đường dẫn đến ảnh phân đoạn (optional)
            metadata: Dict chứa metadata (age, anatomical_site, sex) hoặc các đặc trưng đã xử lý
            alpha: Tỷ lệ kết hợp giữa 2 mô hình (0 đến 1)
            threshold: Ngưỡng xác định lưỡng lự
            
        Returns:
            predicted_class: Lớp dự đoán
            confidence: Độ tin cậy dự đoán
            probs: Xác suất của tất cả các lớp
        """
        # Xử lý ảnh
        img_tensor = self.preprocess_image(image_path, mask_path)
        
        # Xử lý metadata
        meta_tensor = None
        if metadata is not None:
            if isinstance(metadata, dict):
                # Kiểm tra loại metadata
                if 'age' in metadata or 'sex' in metadata or 'anatomical_site' in metadata:
                    # Metadata chưa xử lý
                    patient_metadata = self.process_patient_metadata(
                        age=metadata.get('age'),
                        anatomical_site=metadata.get('anatomical_site'),
                        sex=metadata.get('sex')
                    )
                    meta_tensor = self.preprocess_metadata(patient_metadata)
                else:
                    # Metadata đã xử lý
                    meta_tensor = self.preprocess_metadata(metadata)
        
        # Đưa qua mô hình 9-lớp
        self.model_9class.eval()
        with torch.no_grad():
            outputs_9 = self.model_9class(img_tensor, meta_tensor)
            probs_9 = F.softmax(outputs_9, dim=1).cpu().numpy()[0]
            print(f"Xác suất từ mô hình 9 lớp: {probs_9}")
        final_probs = probs_9
        # Kiểm tra xem dự đoán có thuộc về BKL, MEL, NV không
        if self.is_bkl_mel_nv_prediction(probs_9, threshold):
            self.model_3class.eval()
            with torch.no_grad():
                outputs_3 = self.model_3class(img_tensor, meta_tensor)
                probs_3 = F.softmax(outputs_3, dim=1).cpu().numpy()[0]
                print(f"Xác suất từ mô hình 3 lớp: {probs_3}")
            final_probs = np.copy(probs_9)
            final_probs[self.bkl_idx] = alpha * probs_9[self.bkl_idx] + (1 - alpha) * probs_3[0]
            final_probs[self.mel_idx] = alpha * probs_9[self.mel_idx] + (1 - alpha) * probs_3[1]
            final_probs[self.nv_idx] = alpha * probs_9[self.nv_idx] + (1 - alpha) * probs_3[2]

            # --- Áp dụng meta-classifier nếu có ---
            if self.meta_classifier is not None:
                features = list(probs_9) + list(probs_3)
                if meta_tensor is not None:
                    features += list(meta_tensor.cpu().numpy()[0])
                # Lấy xác suất từ meta-classifier
                meta_probs = self.meta_classifier.predict_proba([features])[0]
                final_probs = meta_probs  # Sử dụng xác suất thật sự
            else:
                final_probs = probs_9

        predicted_idx = np.argmax(final_probs)
        predicted_class = self.class_names[predicted_idx]
        confidence = final_probs[predicted_idx]

        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {cls: float(final_probs[i]) for i, cls in enumerate(self.class_names)}
        }

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn tới các model checkpoint
    MODEL_9CLASS_PATH = "checkpoints/vit_meta_multiclass_tokenmix_best.pth"
    MODEL_3CLASS_PATH = "checkpoints/ensemble_model.pth"  # hoặc model ViT 3-lớp
    
    # Tên các trường metadata
    meta_features = ['age_approx'] + [f'site_{i}' for i in range(9)] + [f'sex_{i}' for i in range(3)]
    
    # Khởi tạo classifier
    classifier = SkinLesionClassifier(
        model_9class_path=MODEL_9CLASS_PATH,
        model_3class_path=MODEL_3CLASS_PATH,
        meta_feature_names=meta_features,
        use_seg=True,
        download_backbone=True
    )
    
    # Phân loại một ảnh
    result = classifier.classify(
        image_path='sample_image.jpg',
        mask_path='sample_mask.png',  # Optional
        metadata={
            'age': 45,
            'anatomical_site': 'head/neck',
            'sex': 'male'
        }
    )
    
    logger.info("\nKết quả phân loại:")
    logger.info(f"Lớp dự đoán: {result['predicted_class']}")
    logger.info(f"Độ tin cậy: {result['confidence']:.4f}")
    logger.info("\nXác suất các lớp:")
    for cls, prob in result['probabilities'].items():
        logger.info(f"{cls}: {prob:.4f}")