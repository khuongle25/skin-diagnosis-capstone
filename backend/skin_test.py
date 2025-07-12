import os
import sys
import argparse
from skin_analyzer import SkinImageAnalyzer, SkinImageType

def main():
    """
    Chương trình chính để kiểm thử SkinImageAnalyzer
    """
    # Tạo parser cho tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Kiểm tra ảnh da")
    parser.add_argument("image_path", help="Đường dẫn đến ảnh cần phân tích")
    parser.add_argument("--model", help="Đường dẫn đến model CNN (tùy chọn)", default=None)
    parser.add_argument("--seg-model", help="Đường dẫn đến model segmentation", 
                      default="checkpoints/skin_full.pth")
    args = parser.parse_args()
    
    # Kiểm tra file tồn tại
    if not os.path.exists(args.image_path):
        print(f"Lỗi: Không tìm thấy file '{args.image_path}'")
        return 1
    
    # Khởi tạo analyzer với mô hình segmentation
    analyzer = SkinImageAnalyzer(
        model_path=args.model,
        segmentation_model_path=args.seg_model
    )
    
    # Phân tích ảnh với chế độ verbose
    result = analyzer.analyze_image(args.image_path, verbose=True)
    
    # Hiển thị kết quả rõ ràng
    print("\n" + "="*50)
    print("KẾT QUẢ PHÂN LOẠI ẢNH DA")
    print("="*50)
    
    if result == SkinImageType.SKIN_CLOSEUP:
        print("✅ ẢNH DA ĐẠT CHUẨN")
    elif result == SkinImageType.SKIN_NONSPECIFIC:
        print("⚠️ ẢNH DA KHÔNG CỤ THỂ") 
    else:
        print("❌ KHÔNG PHẢI ẢNH DA")
    
    print(f"Phân loại: {result.name}")
    print(f"Mô tả: {result.value}")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())