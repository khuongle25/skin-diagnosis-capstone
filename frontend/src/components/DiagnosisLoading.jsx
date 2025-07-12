// src/components/DiagnosisLoading.jsx
import React, { useEffect, useState } from 'react';
import { 
  Box, Paper, Typography, LinearProgress, 
  Stepper, Step, StepLabel, Grid, Fade 
} from '@mui/material';

// Icons
import ScannerIcon from '@mui/icons-material/Scanner';
import BiotechIcon from '@mui/icons-material/Biotech';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';

const steps = [
  {
    label: 'Phân tích ảnh',
    description: 'Đang xử lý ảnh đầu vào và định vị tổn thương...',
    icon: <ScannerIcon />,
    time: 1500
  },
  {
    label: 'Phân đoạn tổn thương',
    description: 'Đang xác định và phân tách khu vực tổn thương trên da...',
    icon: <BiotechIcon />,
    time: 2500
  },
  {
    label: 'Chẩn đoán',
    description: 'Đang phân tích đặc điểm và chẩn đoán loại tổn thương...',
    icon: <MedicalServicesIcon />,
    time: 3500
  }
];

const DiagnosisLoading = ({ hasMetadata = false }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [facts, setFacts] = useState([
    "Ung thư da melanoma là loại ung thư da phổ biến thứ 3 tại Việt Nam",
    "Phát hiện sớm và điều trị kịp thời có thể tăng tỷ lệ sống sót lên tới 99%",
    "Hơn 90% các trường hợp ung thư da có thể phòng ngừa được",
    "Nguyên nhân hàng đầu gây ung thư da là do tia UV từ ánh nắng mặt trời",
    "Việc sử dụng kem chống nắng hàng ngày giảm 50% nguy cơ mắc ung thư da",
    "5-8% tổn thương da đáng ngờ có thể là ác tính",
    "Trí tuệ nhân tạo hiện có thể phát hiện ung thư da với độ chính xác tương đương bác sĩ da liễu",
    "Nên khám da ít nhất mỗi năm một lần",
    "Quy tắc ABCDE giúp nhận biết tổn thương nguy hiểm: Asymmetry, Border, Color, Diameter, Evolving"
  ]);
  const [currentFact, setCurrentFact] = useState(0);

  useEffect(() => {
    // Tạo animation chuyển qua các bước
    const timer = setInterval(() => {
      setProgress((prevProgress) => {
        const nextProgress = prevProgress + 1;
        if (nextProgress >= 100) {
          return 100;
        }
        return nextProgress;
      });
    }, 150);

    // Chuyển sang fact tiếp theo mỗi 4s
    const factTimer = setInterval(() => {
      setCurrentFact((prev) => (prev + 1) % facts.length);
    }, 4000);

    // Chuyển sang bước tiếp theo
    const stepTimer = setTimeout(() => {
      if (activeStep < steps.length - 1) {
        setActiveStep(activeStep + 1);
        setProgress(0);
      }
    }, steps[activeStep].time);

    return () => {
      clearInterval(timer);
      clearInterval(factTimer);
      clearTimeout(stepTimer);
    };
  }, [activeStep, facts.length]);

  // Hiệu ứng nhấp nháy khi đang thực hiện bước cuối
  const [pulse, setPulse] = useState(false);
  useEffect(() => {
    if (activeStep === steps.length - 1) {
      const pulseTimer = setInterval(() => {
        setPulse(prev => !prev);
      }, 800);
      
      return () => clearInterval(pulseTimer);
    }
  }, [activeStep]);

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 4, 
        borderRadius: 2, 
        my: 3, 
        backgroundColor: 'rgba(255,255,255,0.95)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
      }}
    >
      <Typography variant="h4" align="center" gutterBottom sx={{ mb: 3, color: 'primary.main', fontWeight: 'medium' }}>
        Đang phân tích tổn thương da
      </Typography>

      <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 4 }}>
        {steps.map((step, index) => (
          <Step key={step.label}>
            <StepLabel 
              StepIconProps={{ 
                icon: step.icon,
                sx: index === activeStep ? 
                  { 
                    transform: pulse ? 'scale(1.2)' : 'scale(1)',
                    transition: 'transform 0.4s ease-in-out',
                    color: 'primary.main' 
                  } : {}
              }}
            >
              {step.label}
            </StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box sx={{ mb: 4 }}>
        <Typography 
          variant="h6" 
          sx={{ 
            mb: 1.5, 
            color: 'primary.dark',
            animation: activeStep === steps.length - 1 ? 'pulse 1.5s infinite' : 'none',
            '@keyframes pulse': {
              '0%': { opacity: 0.7 },
              '50%': { opacity: 1 },
              '100%': { opacity: 0.7 }
            }
          }}
        >
          {steps[activeStep].label}
        </Typography>
        <Typography variant="body1" sx={{ mb: 2 }}>
          {steps[activeStep].description}
        </Typography>
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          sx={{ 
            height: 8, 
            borderRadius: 4,
            '& .MuiLinearProgress-bar': {
              backgroundImage: 'linear-gradient(to right, #3f51b5, #6573c3)'
            }
          }}
        />
        <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'right' }}>
          {progress}%
        </Typography>
      </Box>

      {hasMetadata && (
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            mb: 3
          }}
        >
          <Box 
            sx={{ 
              px: 3, 
              py: 1, 
              bgcolor: 'success.light', 
              color: 'white', 
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center'
            }}
          >
            <Typography variant="body2">
              Đang sử dụng thông tin bệnh nhân để cải thiện độ chính xác
            </Typography>
          </Box>
        </Box>
      )}

      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12} md={10}>
          <Paper 
            elevation={1}
            sx={{ 
              p: 2, 
              bgcolor: 'info.light', 
              color: 'info.contrastText',
              borderRadius: 2,
              minHeight: 100,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <Fade in={true} timeout={1000} key={currentFact}>
              <Typography variant="body1" align="center">
                <strong>Bạn có biết?</strong><br />
                {facts[currentFact]}
              </Typography>
            </Fade>
          </Paper>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default DiagnosisLoading;