import React, { useState, useEffect } from 'react';
import { 
  Box, Paper, Typography, Button, Grid, TextField, MenuItem, 
  FormControl, FormLabel, RadioGroup, FormControlLabel, Radio
} from '@mui/material';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import EditIcon from '@mui/icons-material/Edit';
import VisibilityIcon from '@mui/icons-material/Visibility';
import LockIcon from '@mui/icons-material/Lock';
import Switch from '@mui/material/Switch';

const locationOptions = [
  { value: 'abdomen', label: 'Bụng' },
  { value: 'back', label: 'Lưng' },
  { value: 'chest', label: 'Ngực' },
  { value: 'ear', label: 'Tai' },
  { value: 'face', label: 'Mặt' },
  { value: 'foot', label: 'Chân' },
  { value: 'genital', label: 'Vùng sinh dục' },
  { value: 'hand', label: 'Tay' },
  { value: 'lower extremity', label: 'Chi dưới' },
  { value: 'neck', label: 'Cổ' },
  { value: 'scalp', label: 'Da đầu' },
  { value: 'trunk', label: 'Thân' },
  { value: 'upper extremity', label: 'Chi trên' },
  { value: 'unknown', label: 'Không rõ' }
];

const ImagePreview = ({ imageSrc, onDiagnose, onCancel, patientData, onUpdateMetadata }) => {
  const [isPublic, setIsPublic] = useState(true);
  
  const handlePublicChange = (event) => {
    setIsPublic(event.target.checked);
  };
  
  // Uncomment và sử dụng hàm handleDiagnose
  const handleDiagnose = () => {
    onDiagnose(isPublic);  // Truyền tham số isPublic
  };
  
  // Bắt đầu với dữ liệu hiện có hoặc khởi tạo mới
  const [formData, setFormData] = useState({
    age: patientData?.age || '',
    gender: patientData?.gender || '',
    location: patientData?.location || ''
  });
  
  // Cập nhật formData khi patientData thay đổi từ bên ngoài
  useEffect(() => {
    if (patientData) {
      setFormData({
        age: patientData.age || '',
        gender: patientData.gender || '',
        location: patientData.location || ''
      });
    }
  }, [patientData]);

  // Xử lý thay đổi trong form
  const handleChange = (e) => {
    const { name, value } = e.target;
    const newFormData = {
      ...formData,
      [name]: value
    };
    setFormData(newFormData);
    
    // Gửi dữ liệu cập nhật ra ngoài
    const hasData = 
      newFormData.age !== '' || 
      newFormData.gender !== '' || 
      newFormData.location !== '';
      
    onUpdateMetadata(hasData ? newFormData : null);
  };

  // Hiển thị dạng cho người dùng
  const getDisplayValue = (key, value) => {
    if (key === 'gender') {
      return value === 'male' ? 'Nam' : 
             value === 'female' ? 'Nữ' : 
             'Không xác định';
    }
    
    if (key === 'location') {
      const locationMap = {
        'anterior torso': 'Vùng bụng trước',
        'head/neck': 'Đầu/Cổ',
        'lateral torso': 'Vùng thân bên',
        'lower extremity': 'Chi dưới',
        'oral/genital': 'Vùng miệng/sinh dục',
        'palms/soles': 'Lòng bàn tay/bàn chân',
        'posterior torso': 'Vùng lưng',
        'upper extremity': 'Chi trên',
        'unknown': 'Không rõ'
      };
      return locationMap[value] || 'Không xác định';
    }
    
    return value;
  };

  return (
    <Paper elevation={3} sx={{ p: 3, borderRadius: 2, my: 3 }}>
      <Typography variant="h5" sx={{ mb: 3, textAlign: 'center', color: 'primary.main' }}>
        Xác nhận và chẩn đoán
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <Paper elevation={1} sx={{ p: 2 }}>
            <Typography variant="subtitle1" sx={{ mb: 1, textAlign: 'center' }}>
              Ảnh đã tải lên
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <img
                src={imageSrc}
                alt="Uploaded"
                style={{ maxWidth: '100%', maxHeight: '350px', objectFit: 'contain' }}
              />
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={5}>
          <Paper elevation={1} sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6" color="primary.main">
                Thông tin bệnh nhân
              </Typography>
              <EditIcon color="action" fontSize="small" />
            </Box>
            
            {/* Form nhập liệu - Luôn hiển thị cho phép chỉnh sửa */}
            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                label="Tuổi"
                name="age"
                type="number"
                value={formData.age}
                onChange={handleChange}
                sx={{ mb: 2 }}
                InputProps={{ inputProps: { min: 0, max: 120 } }}
                placeholder="Nhập tuổi của bệnh nhân"
              />
              
              <FormControl component="fieldset" sx={{ mb: 2, width: '100%' }}>
                <FormLabel component="legend">Giới tính</FormLabel>
                <RadioGroup
                  row
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                >
                  <FormControlLabel value="male" control={<Radio />} label="Nam" />
                  <FormControlLabel value="female" control={<Radio />} label="Nữ" />
                  <FormControlLabel value="unknown" control={<Radio />} label="Không xác định" />
                </RadioGroup>
              </FormControl>
              
              <TextField
                fullWidth
                select
                label="Vị trí tổn thương"
                name="location"
                value={formData.location}
                onChange={handleChange}
                sx={{ mb: 1 }}
                placeholder="Chọn vị trí tổn thương"
              >
                {locationOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </TextField>
            </Box>
            
            <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', mb: 2 }}>
              Thông tin bệnh nhân sẽ giúp cải thiện độ chính xác trong chẩn đoán.
            </Typography>
            {/* <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={isPublic}
                    onChange={handlePublicChange}
                    color="primary"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {isPublic ? <VisibilityIcon sx={{ mr: 0.5 }} /> : <LockIcon sx={{ mr: 0.5 }} />}
                    <Typography variant="body2">
                      {isPublic ? "Công khai" : "Riêng tư"}
                    </Typography>
                  </Box>
                }
              />
            </Box> */}
            <Box sx={{ mt: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                startIcon={<LocalHospitalIcon />}
                onClick={handleDiagnose}
                fullWidth
              >
                Tiến hành chẩn đoán
              </Button>
              
              <Button
                variant="outlined"
                color="secondary"
                onClick={onCancel}
                fullWidth
              >
                Hủy và tải ảnh khác
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default ImagePreview;