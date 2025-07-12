// src/components/ResultDisplay.jsx
import React, { useEffect, useState, useRef } from 'react';
import { 
  Box, Typography, Paper, Grid, Button, CircularProgress, Chip,
  Table, TableBody, TableCell, TableContainer, TableRow  
} from '@mui/material';
import ChatIcon from "@mui/icons-material/Chat";
import MedicalInformationIcon from '@mui/icons-material/MedicalInformation';
import ChatInterface from "./ChatInterface";
// Thêm các import cần thiết
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ReportProblemOutlinedIcon from '@mui/icons-material/ReportProblemOutlined';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ShareIcon from '@mui/icons-material/Share';
import ShareDiagnosisDialog from './ShareDiagnosisDialog';
import { Divider, Fade, Grow } from '@mui/material';
import getLesionDescription from '../services/LesionDescriptions';

const ResultDisplay = ({ result, loading, onReset, showChatbot, onOpenChatbot, onCloseChatbot, currentConversationId, onVisualizeDone }) => {
  const [combinedImage, setCombinedImage] = useState(null);
  const [animatedImage, setAnimatedImage] = useState(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);

  useEffect(() => {
    // Tạo ảnh kết hợp giữa ảnh gốc và mask khi có cả hai
    setIsAnimating(true); // Ban đầu để trạng thái đang loading
    if (result && result.image && result.mask) {
      createOverlayImage(result.image, result.mask);
    }
    
    // Không cần xử lý cleanup vì không còn animation
    return () => {
      // Vẫn giữ logic cleanup để đảm bảo an toàn
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [result]);

  const handleOpenShareDialog = () => {
    setShareDialogOpen(true);
  };

  const handleCloseShareDialog = () => {
    setShareDialogOpen(false);
  };

  const handleShareSuccess = (postData) => {
    // Có thể thêm logic xử lý sau khi chia sẻ thành công nếu cần
    console.log("Đã chia sẻ thành công:", postData);
  };

  const createOverlayImage = (originalBase64, maskBase64) => {
    const originalImage = new Image();
    const maskImage = new Image();
  
    originalImage.onload = () => {
      maskImage.onload = () => {
        // Chỉ tạo ảnh overlay tĩnh (bỏ phần animation)
        createStaticOverlay(originalImage, maskImage);
        
        // Báo hiệu việc tạo ảnh đã hoàn thành ngay lập tức
        if (onVisualizeDone) {
          setTimeout(() => {
            onVisualizeDone();
            setIsAnimating(false);
          }, 300);
        }
      };
      maskImage.src = `data:image/png;base64,${maskBase64}`;
    };
    originalImage.src = `data:image/png;base64,${originalBase64}`;
  };
  
  // Nâng cấp hàm createStaticOverlay để hiển thị viền tốt hơn
  const createStaticOverlay = (originalImage, maskImage) => {
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
    const maskData = boundaryCtx.getImageData(0, 0, boundaryCanvas.width, boundaryCanvas.height);
    const boundaryPoints = findBoundaryPoints(maskData);
    
    // Sắp xếp những điểm viền
    const sortedPoints = sortBoundaryPoints(boundaryPoints);
    
    // Vẽ viền nét đứt với hiệu ứng nổi bật hơn
    drawEnhancedBorder(ctx, sortedPoints, canvas.width / boundaryCanvas.width, canvas.height / boundaryCanvas.height);
    
    const combinedBase64 = canvas.toDataURL('image/png');
    setCombinedImage(combinedBase64.split(',')[1]);
    // Đặt luôn cho animated image để UI hiển thị đúng
    setAnimatedImage(combinedBase64.split(',')[1]);
  }
  
  // Thêm hàm vẽ viền nổi bật hơn
  const drawEnhancedBorder = (ctx, points, scaleX, scaleY) => {
    if (points.length === 0) return;
    
    // Vẽ viền với glow effect
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
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash([8, 4]); // Nét đứt
    ctx.shadowColor = 'rgba(255, 255, 255, 0.8)';
    ctx.shadowBlur = 15;
    ctx.stroke();
    
    // Vẽ lại với đường viền mỏng hơn để tạo hiệu ứng sắc nét
    ctx.strokeStyle = 'rgba(255, 0, 0, 1)';
    ctx.lineWidth = 1.5;
    ctx.shadowBlur = 5;
    ctx.stroke();
    
    ctx.restore();
  };

  const drawStaticBorder = (ctx, boundaryPoints, scaleX, scaleY) => {
    if (boundaryPoints.length === 0) return;
    
    ctx.save();
    
    // Cài đặt style cho viền nét đứt tĩnh
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash([16, 9]); // Nét đứt: 12px nét, 6px khoảng trống
    ctx.shadowColor = 'rgba(255, 255, 255, 0.6)';
    ctx.shadowBlur = 30;
    
    // Vẽ đường viền theo boundary points
    ctx.beginPath();
    
    for (let i = 0; i < boundaryPoints.length; i++) {
      const point = boundaryPoints[i];
      if (i === 0) {
        ctx.moveTo(point.x * scaleX, point.y * scaleY);
      } else {
        ctx.lineTo(point.x * scaleX, point.y * scaleY);
      }
    }
    
    // Đóng đường viền
    if (boundaryPoints.length > 0) {
      const firstPoint = boundaryPoints[0];
      ctx.lineTo(firstPoint.x * scaleX, firstPoint.y * scaleY);
    }
    
    ctx.stroke();
    ctx.restore();
  };

  const createAnimatedOverlay = (originalImage, maskImage) => {
    const canvas = document.createElement('canvas');
    canvas.width = originalImage.width;
    canvas.height = originalImage.height;
    const ctx = canvas.getContext('2d');
  
    // Tạo mask boundary để vẽ hiệu ứng viền
    const boundaryCanvas = document.createElement('canvas');
    boundaryCanvas.width = maskImage.width;
    boundaryCanvas.height = maskImage.height;
    const boundaryCtx = boundaryCanvas.getContext('2d');
    
    // Vẽ mask lên boundary canvas
    boundaryCtx.drawImage(maskImage, 0, 0);
    const maskData = boundaryCtx.getImageData(0, 0, boundaryCanvas.width, boundaryCanvas.height);
    
    // Tìm boundary points (viền của mask)
    const boundaryPoints = findBoundaryPoints(maskData);
    
    let animationFrame = 0;
    setIsAnimating(true);
  
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Vẽ ảnh gốc
      ctx.drawImage(originalImage, 0, 0);
      
      // Vẽ viền nét đứt chạy
      drawAnimatedBorder(ctx, boundaryPoints, animationFrame, canvas.width / boundaryCanvas.width, canvas.height / boundaryCanvas.height);
      
      animationFrame++;
      
      // Chạy vô hạn để có hiệu ứng liên tục
      animationRef.current = requestAnimationFrame(animate);
      
      // Update state với ảnh hiện tại
      const currentBase64 = canvas.toDataURL('image/png');
      setAnimatedImage(currentBase64.split(',')[1]);
      
      // Gọi callback sau 2 giây để báo animation "hoàn thành" (nhưng vẫn tiếp tục chạy)
      if (animationFrame === 120 && onVisualizeDone) {
        onVisualizeDone();
        setIsAnimating(false);
      }
    };
  
    animate();
  };

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
    
    return boundaryPoints;
  };

  const drawAnimatedBorder = (ctx, boundaryPoints, frame, scaleX, scaleY) => {
    if (boundaryPoints.length === 0) return;
    
    ctx.save();
    
    // Sắp xếp boundary points theo thứ tự để tạo thành đường viền liên tục
    const sortedPoints = sortBoundaryPoints(boundaryPoints);
    
    // Tính toán offset cho hiệu ứng "marching ants"
    const dashLength = 8;
    const gapLength = 4;
    const totalDashPattern = dashLength + gapLength;
    const animationOffset = (frame * 0.5) % totalDashPattern;
    
    // Vẽ viền nét đứt chạy (marching ants effect)
    ctx.strokeStyle = 'rgba(225, 13, 13, 0.9)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(255, 255, 255, 0.6)';
    ctx.shadowBlur = 3;
    
    // Thiết lập nét đứt với offset animation
    ctx.setLineDash([dashLength, gapLength]);
    ctx.lineDashOffset = -animationOffset;
    
    // Vẽ đường viền chính
    ctx.beginPath();
    
    if (sortedPoints.length > 0) {
      const firstPoint = sortedPoints[0];
      ctx.moveTo(firstPoint.x * scaleX, firstPoint.y * scaleY);
      
      for (let i = 1; i < sortedPoints.length; i++) {
        const point = sortedPoints[i];
        ctx.lineTo(point.x * scaleX, point.y * scaleY);
      }
      
      // Đóng đường viền
      ctx.closePath();
    }
    
    ctx.stroke();
    
    // Thêm hiệu ứng glow bằng cách vẽ thêm lớp mờ hơn
    const glowIntensity = 0.3 + 0.2 * Math.sin(frame * 0.1);
    ctx.strokeStyle = `rgba(255, 0, 0, ${glowIntensity})`;
    ctx.lineWidth = 4;
    ctx.shadowBlur = 15;
    ctx.setLineDash([dashLength, gapLength]);
    ctx.lineDashOffset = -animationOffset;
    
    ctx.stroke();
    
    ctx.restore();
  };
  
  // Hàm sắp xếp boundary points để tạo thành đường viền liên tục
  const sortBoundaryPoints = (points) => {
    if (points.length === 0) return [];
    
    const sortedPoints = [];
    const visited = new Set();
    
    // Bắt đầu từ điểm có y nhỏ nhất (điểm trên cùng)
    let currentPoint = points.reduce((min, point) => 
      point.y < min.y || (point.y === min.y && point.x < min.x) ? point : min
    );
    
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

  if (!result) return null;
  const lesionType = result?.diagnosis?.label;
  const lesionFullName = getLesionDescription(lesionType)?.name || lesionType;

  return (
    <Grid container spacing={3} sx={{ mt: 4 }}>
      {/* Diagnosis section */}
      <Grid item xs={12} md={showChatbot ? 7 : 12}>
        <Paper elevation={3} sx={{ p: 3, borderRadius: 2, my: 3, height: '100%' }}>
        {result.diagnosis && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 3, p: 2, bgcolor: 'background.paper', borderRadius: 2, border: '1px solid #e0e0e0' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, gap: 1 }}>
              <MedicalInformationIcon color="primary" sx={{ fontSize: 28 }} />
              <Typography variant="h6" color="primary.main">
                Kết quả chẩn đoán
              </Typography>
            </Box>
            
            {/* Hiển thị tên đầy đủ thay vì tên viết tắt */}
            <Typography variant="h5" sx={{ fontWeight: 'bold', my: 1 }}>
              {getLesionDescription(result.diagnosis.label)?.name || result.diagnosis.label}
            </Typography>
            
            <Chip 
              label={`Độ tin cậy: ${(result.diagnosis.confidence * 100).toFixed(2)}%`}
              color="primary" 
              variant="outlined"
              sx={{ mb: 2 }}
            />
            
            {/* Top 3 kết quả có xác suất cao nhất - Hiển thị tên đầy đủ */}
            {result.diagnosis.top_predictions && (
              <TableContainer component={Paper} elevation={0} sx={{ maxWidth: 500, mt: 2 }}>
                <Table size="small">
                  <TableBody>
                    {result.diagnosis.top_predictions.map((pred, index) => (
                      <TableRow key={index} sx={index === 0 ? { bgcolor: 'rgba(63, 81, 181, 0.1)' } : {}}>
                        <TableCell component="th" scope="row" sx={{ fontWeight: index === 0 ? 'bold' : 'normal' }}>
                          {/* Chuyển tên viết tắt thành tên đầy đủ */}
                          {getLesionDescription(pred.label)?.name || pred.label}
                        </TableCell>
                        <TableCell align="right" sx={{ fontWeight: index === 0 ? 'bold' : 'normal' }}>
                          {(pred.confidence * 100).toFixed(2)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
            
            {result.metadata_used ? (
              <Chip 
                sx={{ mt: 2 }}
                label="Chẩn đoán có sử dụng thông tin bệnh nhân" 
                color="success" 
                size="small"
              />
            ) : (
              <Chip 
                sx={{ mt: 2 }}
                label="Chẩn đoán không sử dụng thông tin bệnh nhân" 
                color="default" 
                size="small"
              />
            )}
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
              Đây là kết quả chẩn đoán từ AI và cần được xác nhận bởi chuyên gia y tế.
            </Typography>
          </Box>
        )}

    <Grid container spacing={3} sx={{ mt: 2 }}>
      <Grid item xs={12}>
        <Paper elevation={2} sx={{ p: 2, height: '100%', position: 'relative' }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold', textAlign: 'center' }}>
            {isAnimating ? "Đang phân tích tổn thương..." : "Chi tiết phân tích tổn thương"}
          </Typography>
          
          <Box sx={{ 
            display: 'flex', 
            flexDirection: { xs: 'column', md: 'row' },
            gap: 3
          }}>
            {/* Phần bên trái: Hình ảnh phân đoạn */}
            <Box sx={{ 
              flex: 1, 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {animatedImage ? (
                <Grow in={!isAnimating} timeout={800}>
                  <img 
                    src={`data:image/png;base64,${animatedImage}`}
                    alt="Animated Segmentation" 
                    style={{ 
                      width: '100%', 
                      maxHeight: '400px', 
                      objectFit: 'contain',
                      borderRadius: '8px',
                      boxShadow: isAnimating ? '0 0 20px rgba(255, 255, 255, 0.3)' : '0 4px 8px rgba(0, 0, 0, 0.1)'
                    }} 
                  />
                </Grow>
              ) : combinedImage ? (
                <img 
                  src={`data:image/png;base64,${combinedImage}`}
                  alt="Static Segmentation" 
                  style={{ width: '100%', maxHeight: '400px', objectFit: 'contain', borderRadius: '8px' }} 
                />
              ) : (
                <img 
                  src={`data:image/png;base64,${result.image}`}
                  alt="Original" 
                  style={{ width: '100%', maxHeight: '400px', objectFit: 'contain', borderRadius: '8px' }} 
                />
              )}
              
              <Typography variant="subtitle2" sx={{ mt: 2, color: 'text.secondary', fontStyle: 'italic', textAlign: 'center' }}>
                {isAnimating ? "Hệ thống đang phân tích tổn thương..." : "Vùng tổn thương được xác định"}
              </Typography>
            </Box>
            
            {/* Phần bên phải: Thông tin tổn thương */}
            <Fade in={!isAnimating} timeout={1200}>
              <Box sx={{ 
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                borderLeft: { xs: 'none', md: '1px solid #e0e0e0' },
                paddingLeft: { xs: 0, md: 3 },
                paddingTop: { xs: 3, md: 0 },
                borderTop: { xs: '1px solid #e0e0e0', md: 'none' },
                marginTop: { xs: 2, md: 0 }
              }}>
                {result.diagnosis && getLesionDescription(result.diagnosis.label) ? (
                  <>
                    <Box sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      mb: 2,
                      p: 1,
                      borderRadius: '4px',
                      bgcolor: getLesionDescription(result.diagnosis.label).severity === 'high'
                        ? 'rgba(244, 67, 54, 0.08)'
                        : getLesionDescription(result.diagnosis.label).severity === 'medium'
                          ? 'rgba(255, 152, 0, 0.08)'
                          : 'rgba(76, 175, 80, 0.08)'
                    }}>
                      {getLesionDescription(result.diagnosis.label).severity === 'high' ? (
                        <WarningIcon sx={{ color: '#f44336', fontSize: 28, mr: 1.5 }} />
                      ) : getLesionDescription(result.diagnosis.label).severity === 'medium' ? (
                        <ReportProblemOutlinedIcon sx={{ color: '#ff9800', fontSize: 28, mr: 1.5 }} />
                      ) : (
                        <CheckCircleOutlineIcon sx={{ color: '#4caf50', fontSize: 28, mr: 1.5 }} />
                      )}
                      
                      <Typography variant="h6" sx={{ 
                        fontWeight: 500,
                        color: getLesionDescription(result.diagnosis.label).severity === 'high' 
                          ? '#d32f2f' 
                          : getLesionDescription(result.diagnosis.label).severity === 'medium'
                            ? '#e65100'
                            : '#2e7d32'
                      }}>
                        {getLesionDescription(result.diagnosis.label).name}
                      </Typography>
                    </Box>
                    
                    <Grow in={!isAnimating} timeout={1500}>
                      <Box sx={{ 
                        p: 2, 
                        borderRadius: '4px', 
                        bgcolor: 'rgba(0, 0, 0, 0.02)',
                        border: '1px dashed rgba(0, 0, 0, 0.15)',
                        mb: 2
                      }}>
                        <Typography variant="body1" sx={{ 
                          fontSize: '16px',
                          lineHeight: 1.6,
                          textAlign: 'justify' 
                        }}>
                          {getLesionDescription(result.diagnosis.label).description}
                        </Typography>
                      </Box>
                    </Grow>
                    
                    <Divider sx={{ my: 2 }} />
                    
                    {/* Thông tin mức độ nghiêm trọng */}
                    <Grow in={!isAnimating} timeout={1800}>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1, display: 'flex', alignItems: 'center' }}>
                          <InfoIcon fontSize="small" sx={{ mr: 1, color: 'primary.main' }} />
                          Mức độ:
                          <Box component="span" sx={{ 
                            ml: 1,
                            fontWeight: 'normal',
                            color: getLesionDescription(result.diagnosis.label).severity === 'high'
                              ? '#d32f2f'
                              : getLesionDescription(result.diagnosis.label).severity === 'medium'
                                ? '#e65100'
                                : '#2e7d32'
                          }}>
                            {getLesionDescription(result.diagnosis.label).severity === 'high'
                              ? 'Cao (cần thăm khám gấp)'
                              : getLesionDescription(result.diagnosis.label).severity === 'medium'
                                ? 'Trung bình (cần theo dõi)'
                                : 'Thấp (lành tính)'}
                          </Box>
                        </Typography>
                      </Box>
                    </Grow>
                    
                    {/* Thông tin độ tin cậy */}
                    <Grow in={!isAnimating} timeout={2000}>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1, display: 'flex', alignItems: 'center' }}>
                          <InfoIcon fontSize="small" sx={{ mr: 1, color: 'primary.main' }} />
                          Độ tin cậy: 
                          <Box component="span" sx={{ ml: 1, fontWeight: 'normal' }}>
                            {(result.diagnosis.confidence * 100).toFixed(2)}%
                          </Box>
                        </Typography>
                        
                        {/* Thanh độ tin cậy */}
                        <Box sx={{ 
                          width: '100%', 
                          height: 8, 
                          bgcolor: 'rgba(0,0,0,0.1)', 
                          borderRadius: 5,
                          position: 'relative',
                          overflow: 'hidden'
                        }}>
                          <Box 
                            sx={{ 
                              width: `${result.diagnosis.confidence * 100}%`, 
                              height: '100%',
                              bgcolor: result.diagnosis.confidence > 0.7 
                                ? 'success.main' 
                                : result.diagnosis.confidence > 0.5 
                                  ? 'warning.main' 
                                  : 'error.main',
                              transition: 'width 1.5s ease-in-out'
                            }} 
                          />
                        </Box>
                      </Box>
                    </Grow>
                    
                    {/* Lưu ý */}
                    <Fade in={!isAnimating} timeout={2200}>
                      <Box sx={{ 
                        mt: 'auto',
                        p: 1.5, 
                        borderRadius: '4px', 
                        bgcolor: 'primary.light', 
                        color: 'white',
                        display: 'flex',
                        alignItems: 'flex-start'
                      }}>
                        <InfoIcon sx={{ mr: 1, mt: 0.2, fontSize: '1.2rem' }} />
                        <Typography variant="body2">
                          Đây là kết quả phân tích từ AI. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để được tư vấn chi tiết.
                        </Typography>
                      </Box>
                    </Fade>
                  </>
                ) : (
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center',
                    justifyContent: 'center', 
                    height: '100%', 
                    p: 3 
                  }}>
                    <HelpOutlineIcon sx={{ fontSize: 60, color: '#9e9e9e', mb: 2 }} />
                    <Typography variant="body1" sx={{ textAlign: 'center', color: 'text.secondary' }}>
                      Không có thông tin chi tiết về tổn thương này
                    </Typography>
                  </Box>
                )}
              </Box>
            </Fade>
          </Box>
        </Paper>
      </Grid>
    </Grid>

          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3, gap: 2 }}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={onReset}
              sx={{ px: 4, py: 1 }}
            >
              Tải ảnh khác
            </Button>
            <Button
              variant="contained"
              onClick={handleOpenShareDialog}
              startIcon={<ShareIcon />}
              sx={{
                px: 4,
                py: 1,
                bgcolor: 'success.main',
                '&:hover': { bgcolor: 'success.dark' }
              }}
              disabled={isAnimating} // Disable button khi đang animation
            >
              Chia sẻ
            </Button>
            {lesionType && (
              <Button 
                variant="contained"
                onClick={showChatbot ? onCloseChatbot : () => onOpenChatbot(lesionFullName)} 
                startIcon={<ChatIcon />}
                sx={{ 
                  px: 4, 
                  py: 1, 
                  bgcolor: 'purple', 
                  '&:hover': { bgcolor: 'darkpurple' } 
                }}
                disabled={isAnimating} // Disable button khi đang animation
              >
                {showChatbot ? "Ẩn chatbot" : "Chatbot tư vấn"}
              </Button>
            )}
          </Box>
           {/* Add the dialog component */}
           <ShareDiagnosisDialog
            open={shareDialogOpen}
            onClose={handleCloseShareDialog}
            diagnosisResult={result}
            onSuccess={handleShareSuccess}
          />
        </Paper>
      </Grid>
      
      {/* Chatbot section */}
      {lesionType && showChatbot && !isAnimating && (
        <Grid item xs={12} md={5}>
          <Paper elevation={3} sx={{ 
            p: 0,
            my: 3, 
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            height: '600px',
            maxHeight: '780px',
            overflow: 'hidden',
            borderRadius: 2
          }}>
            <ChatInterface lesionType={lesionFullName} conversationId={currentConversationId}/> 
          </Paper>
        </Grid>
      )}
    </Grid>
  );
};

export default ResultDisplay;