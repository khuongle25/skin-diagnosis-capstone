import React, { useState } from 'react';
import { 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions, 
  TextField,
  Button,
  Box,
  Typography,
  Switch,
  FormControlLabel,
  CircularProgress,
  Snackbar,
  Alert,
  Divider,
  Chip,
  Avatar,
  Paper,
  Fade,
  Zoom,
  IconButton,
  Tooltip
} from '@mui/material';
import ShareIcon from '@mui/icons-material/Share';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import PersonIcon from '@mui/icons-material/Person';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import PhotoIcon from '@mui/icons-material/Photo';
import CloseIcon from '@mui/icons-material/Close';
import InfoIcon from '@mui/icons-material/Info';
import { styled, keyframes } from '@mui/material/styles';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Animations
const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-3px); }
  100% { transform: translateY(0px); }
`;

const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(63, 81, 181, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(63, 81, 181, 0); }
  100% { box-shadow: 0 0 0 0 rgba(63, 81, 181, 0); }
`;

// Styled Components
const StyledDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiDialog-paper': {
    borderRadius: 20,
    overflow: 'hidden',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    position: 'relative',
    '&::before': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
      pointerEvents: 'none',
    }
  },
}));

const StyledDialogTitle = styled(DialogTitle)(({ theme }) => ({
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  color: 'white',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: '-50%',
    left: '-50%',
    width: '200%',
    height: '200%',
    background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%)',
    animation: `${float} 3s ease-in-out infinite`,
  }
}));

const StyledDialogContent = styled(DialogContent)(({ theme }) => ({
  background: 'white',
  position: 'relative',
  padding: theme.spacing(2), // Giảm từ 2.5 xuống 2
}));

const DiagnosisCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
  borderRadius: 15,
  border: '1px solid rgba(103, 126, 234, 0.2)',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 4,
    background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
  }
}));

const AnimatedButton = styled(Button)(({ theme }) => ({
  borderRadius: 25,
  padding: theme.spacing(1.5, 3),
  textTransform: 'none',
  fontWeight: 600,
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 25px rgba(0,0,0,0.15)',
  },
  '&.Mui-disabled': {
    background: theme.palette.grey[300],
  }
}));

const PulseButton = styled(AnimatedButton)(({ theme }) => ({
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  '&:not(.Mui-disabled)': {
    animation: `${pulse} 2s infinite`,
  },
  '&:hover': {
    background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
  }
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: 12,
    transition: 'all 0.3s ease',
    '&:hover': {
      transform: 'translateY(-1px)',
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    },
    '&.Mui-focused': {
      transform: 'translateY(-1px)',
      boxShadow: '0 4px 12px rgba(103, 126, 234, 0.3)',
    }
  }
}));

const ShareDiagnosisDialog = ({ 
  open, 
  onClose, 
  diagnosisResult,
  onSuccess
}) => {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isPublic, setIsPublic] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');
  
  const { user } = useAuth();

  const handleSubmit = async () => {
    if (!user) {
      setSnackbarMessage('Bạn cần đăng nhập để chia sẻ kết quả chẩn đoán');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
      return;
    }
    
    setIsLoading(true);
    
    try {
      const userDisplayName = user.display_name || user.displayName || 'Người dùng ẩn danh';
      
      let patientMetadata = {};
      if (diagnosisResult.metadata) {
        patientMetadata = {
          age: diagnosisResult.metadata.age || null,
          gender: diagnosisResult.metadata.gender || null,
          location: diagnosisResult.metadata.location || null
        };
      }
            
      const postData = {
        user_id: user.uid,
        user_display_name: userDisplayName,
        title: title,
        content: content,
        image_data: diagnosisResult.image,
        mask_data: diagnosisResult.mask,
        patient_metadata: patientMetadata,
        diagnosis: diagnosisResult.diagnosis,
        visible: isPublic
      };
      
      const response = await axios.post(`${API_URL}/api/create-post-from-diagnosis/`, postData);
      
      setSnackbarMessage('Chia sẻ thành công! 🎉');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      
      if (onSuccess && typeof onSuccess === 'function') {
        onSuccess(response.data);
      }
      
      // Reset form
      setTitle('');
      setContent('');
      setIsPublic(true);
      
      setTimeout(() => onClose(), 1000);
    } catch (error) {
      console.error('Lỗi khi chia sẻ:', error);
      setSnackbarMessage(error.response?.data?.detail || 'Có lỗi xảy ra khi chia sẻ');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  const getDiagnosisColor = (label) => {
    const colors = {
      'MEL': '#e53e3e',
      'NV': '#38a169', 
      'BCC': '#dd6b20',
      'AK': '#d69e2e',
      'BKL': '#3182ce',
      'DF': '#805ad5',
      'VASC': '#e53e3e',
      'SCC': '#c53030',
      'UNK': '#718096'
    };
    return colors[label] || '#667eea';
  };

  return (
    <>
      <StyledDialog open={open} onClose={onClose} maxWidth="md" fullWidth>
        <StyledDialogTitle>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            position: 'relative',
            zIndex: 1
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Avatar sx={{ 
                bgcolor: 'rgba(255,255,255,0.2)', 
                mr: 2,
                animation: `${float} 3s ease-in-out infinite`
              }}>
                <ShareIcon />
              </Avatar>
              <Box>
                <Typography variant="h5" component="h2" fontWeight="bold">
                  Chia sẻ kết quả chẩn đoán
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9, mt: 0.5 }}>
                  Chia sẻ với cộng đồng để giúp đỡ lẫn nhau
                </Typography>
              </Box>
            </Box>
            <IconButton 
              onClick={onClose} 
              sx={{ 
                color: 'white',
                '&:hover': { backgroundColor: 'rgba(255,255,255,0.1)' }
              }}
            >
              <CloseIcon />
            </IconButton>
          </Box>
        </StyledDialogTitle>
        
        <StyledDialogContent>
          
          <Fade in={open} timeout={500}>
            <Box>
              <Divider sx={{ my: 3, opacity: 0.3 }} />
              
              {/* Form Fields */}
              <Box sx={{ space: 3 }}>
                <StyledTextField
                  label="✨ Tiêu đề bài chia sẻ"
                  variant="outlined"
                  fullWidth
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  sx={{ mb: 3 }}
                  required
                  placeholder="VD: Kết quả chẩn đoán nốt ruồi trên cánh tay"
                  helperText="Tiêu đề sẽ giúp mọi người hiểu rõ hơn về ca bệnh của bạn"
                />
                
                <StyledTextField
                  label="📝 Nội dung chia sẻ"
                  variant="outlined"
                  fullWidth
                  multiline
                  rows={4}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Chia sẻ câu chuyện, triệu chứng, hoặc cảm nghĩ của bạn về kết quả này. Thông tin của bạn có thể giúp ích cho những người khác có tình trạng tương tự..."
                  required
                  sx={{ mb: 3 }}
                  helperText="Hãy chia sẻ một cách chân thật và hữu ích"
                />
                
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  p: 2,
                  bgcolor: 'grey.50',
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'grey.200'
                }}>
                  <Box>
                    <Typography variant="subtitle1" fontWeight="medium">
                      Quyền riêng tư
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {isPublic ? 'Mọi người có thể xem bài chia sẻ của bạn' : 'Chỉ bạn có thể xem bài này'}
                    </Typography>
                  </Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={isPublic}
                        onChange={(e) => setIsPublic(e.target.checked)}
                        color="primary"
                      />
                    }
                    label={
                      <Box sx={{ display: 'flex', alignItems: 'center', ml: 1 }}>
                        {isPublic ? (
                          <VisibilityIcon color="primary" sx={{ mr: 0.5 }} />
                        ) : (
                          <VisibilityOffIcon color="action" sx={{ mr: 0.5 }} />
                        )}
                        <Typography variant="body2" fontWeight="medium">
                          {isPublic ? 'Công khai' : 'Riêng tư'}
                        </Typography>
                      </Box>
                    }
                  />
                </Box>
              </Box>
            </Box>
          </Fade>
        </StyledDialogContent>
        
        <DialogActions sx={{ px: 3, py: 2, bgcolor: 'grey.50', justifyContent: 'center' }}>
          <Zoom in={!isLoading && title.trim() && content.trim()}>
            <PulseButton 
              variant="contained" 
              onClick={handleSubmit}
              disabled={isLoading || !title.trim() || !content.trim()}
              startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : <ShareIcon />}
              size="large"
              sx={{ 
                minWidth: '200px',
                fontSize: '1rem'
              }}
            >
              {isLoading ? 'Đang chia sẻ...' : 'Chia sẻ ngay 🚀'}
            </PulseButton>
          </Zoom>
        </DialogActions>
      </StyledDialog>
      
      <Snackbar 
        open={snackbarOpen} 
        autoHideDuration={4000} 
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbarSeverity} 
          elevation={6} 
          variant="filled"
          sx={{ 
            borderRadius: 3,
            '& .MuiAlert-message': {
              fontSize: '1rem',
              fontWeight: 500
            }
          }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </>
  );
};

export default ShareDiagnosisDialog;