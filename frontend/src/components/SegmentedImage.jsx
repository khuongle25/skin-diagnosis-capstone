import React, { useEffect, useState } from 'react';
import { Box, CircularProgress } from '@mui/material';

const SegmentedImage = ({ imageData, maskData, height = 240 }) => {
  const [combinedImage, setCombinedImage] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    console.log("Image data:", !!imageData, "Mask data:", !!maskData);
    if (!imageData || !maskData) {
      setLoading(false);
      return;
    }

    const createOverlayImage = () => {
      const originalImage = new Image();
      const maskImage = new Image();
      
      originalImage.onload = () => {
        maskImage.onload = () => {
          // Tạo overlay tĩnh
          const canvas = document.createElement('canvas');
          canvas.width = originalImage.width;
          canvas.height = originalImage.height;
          const ctx = canvas.getContext('2d');
          
          // Vẽ ảnh gốc
          ctx.drawImage(originalImage, 0, 0);
          
          // Tìm boundary points để vẽ viền
          const boundaryCanvas = document.createElement('canvas');
          boundaryCanvas.width = maskImage.width;
          boundaryCanvas.height = maskImage.height;
          const boundaryCtx = boundaryCanvas.getContext('2d');
          
          boundaryCtx.drawImage(maskImage, 0, 0);
          const imageData = boundaryCtx.getImageData(0, 0, boundaryCanvas.width, boundaryCanvas.height);
          
          // Vẽ viền
          drawEnhancedBorder(
            ctx, 
            findBoundaryPoints(imageData), 
            canvas.width / boundaryCanvas.width, 
            canvas.height / boundaryCanvas.height
          );
          
          const base64 = canvas.toDataURL('image/png');
          setCombinedImage(base64);
          setLoading(false);
        };
        
        maskImage.src = `data:image/png;base64,${maskData}`;
      };
      
      originalImage.src = `data:image/png;base64,${imageData}`;
    };
    
    createOverlayImage();
  }, [imageData, maskData]);

  // Hàm tìm các điểm viền của mask
  const findBoundaryPoints = (imageData) => {
    const { data, width, height } = imageData;
    const boundaryPoints = [];
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const currentIndex = (y * width + x) * 4;
        
        if (data[currentIndex] > 100) { // Nếu pixel hiện tại thuộc mask
          // Kiểm tra 8 pixels xung quanh
          let isBoundary = false;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              
              const neighborIndex = ((y + dy) * width + (x + dx)) * 4;
              if (data[neighborIndex] <= 100) { // Nếu có pixel xung quanh không thuộc mask
                isBoundary = true;
                break;
              }
            }
            if (isBoundary) break;
          }
          
          if (isBoundary) {
            boundaryPoints.push({ x, y });
          }
        }
      }
    }
    
    return sortBoundaryPoints(boundaryPoints);
  };

  // Hàm sắp xếp boundary points theo trình tự hợp lý
  const sortBoundaryPoints = (points) => {
    if (points.length === 0) return [];
    
    const sortedPoints = [];
    const visited = new Set();
    
    // Bắt đầu từ điểm có y nhỏ nhất (điểm trên cùng)
    let currentPoint = points.reduce((min, point) => 
      point.y < min.y || (point.y === min.y && point.x < min.x) ? point : min
    , points[0]);
    
    sortedPoints.push(currentPoint);
    visited.add(`${currentPoint.x},${currentPoint.y}`);
    
    // Tìm điểm tiếp theo gần nhất
    while (sortedPoints.length < points.length) {
      let nearestPoint = null;
      let minDistance = Infinity;
      
      for (const point of points) {
        const key = `${point.x},${point.y}`;
        if (!visited.has(key)) {
          const distance = Math.sqrt(
            Math.pow(point.x - currentPoint.x, 2) + 
            Math.pow(point.y - currentPoint.y, 2)
          );
          
          if (distance < minDistance) {
            minDistance = distance;
            nearestPoint = point;
          }
        }
      }
      
      if (nearestPoint && minDistance < 5) { // Chỉ kết nối các điểm gần nhau
        sortedPoints.push(nearestPoint);
        visited.add(`${nearestPoint.x},${nearestPoint.y}`);
        currentPoint = nearestPoint;
      } else {
        break;
      }
    }
    
    return sortedPoints;
  };

  // Hàm vẽ viền nổi bật
  const drawEnhancedBorder = (ctx, points, scaleX, scaleY) => {
    if (points.length === 0) return;
    
    ctx.save();
    
    // Đầu tiên vẽ lớp ngoài với đường nét đậm và mờ (glow)
    ctx.beginPath();
    if (points.length > 0) {
      const firstPoint = points[0];
      ctx.moveTo(firstPoint.x * scaleX, firstPoint.y * scaleY);
      
      for (let i = 1; i < points.length; i++) {
        const point = points[i];
        ctx.lineTo(point.x * scaleX, point.y * scaleY);
      }
      
      // Đóng đường viền
      ctx.closePath();
    }
    
    // Thiết lập style cho viền nổi bật
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash([6, 3]); // Nét đứt
    ctx.shadowColor = 'rgba(255, 255, 255, 0.7)';
    ctx.shadowBlur = 8;
    ctx.stroke();
    
    ctx.restore();
  };

  if (loading) {
    return (
      <Box sx={{ 
        height, 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        bgcolor: 'rgba(0,0,0,0.03)',
        borderRadius: 1
      }}>
        <CircularProgress size={30} />
      </Box>
    );
  }

  return (
    <Box sx={{ 
      height,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      overflow: 'hidden',
      borderRadius: 1,
      border: '1px solid #eee',
      bgcolor: '#fafafa',
      position: 'relative'
    }}>
      <img
        src={combinedImage || `data:image/png;base64,${imageData}`}
        alt="Lesion with segmentation"
        style={{ 
          maxWidth: '100%', 
          maxHeight: '100%', 
          objectFit: 'contain',
          borderRadius: 4
        }}
      />
    </Box>
  );
};

export default SegmentedImage;