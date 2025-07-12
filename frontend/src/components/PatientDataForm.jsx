// src/components/PatientDataForm.jsx - Cập nhật lại phù hợp với metadata model
import React, { useState, useEffect } from 'react';
import { 
  Box, Paper, Typography, TextField, MenuItem, 
  FormControl, FormLabel, RadioGroup, FormControlLabel, 
  Radio
} from '@mui/material';

// Cập nhật locationOptions phù hợp với các giá trị của model
const locationOptions = [
  { value: 'anterior torso', label: 'Vùng bụng trước' },
  { value: 'head/neck', label: 'Đầu/Cổ' },
  { value: 'lateral torso', label: 'Vùng thân bên' },
  { value: 'lower extremity', label: 'Chi dưới' },
  { value: 'oral/genital', label: 'Vùng miệng/sinh dục' },
  { value: 'palms/soles', label: 'Lòng bàn tay/bàn chân' },
  { value: 'posterior torso', label: 'Vùng lưng' },
  { value: 'upper extremity', label: 'Chi trên' },
  { value: 'unknown', label: 'Không rõ' }
];

const PatientDataForm = ({ onDataChange }) => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    location: ''
  });
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    const newFormData = {
      ...formData,
      [name]: value
    };
    setFormData(newFormData);
    
    // Kiểm tra xem có dữ liệu nào được nhập hay không
    const hasData = 
      newFormData.age !== '' || 
      newFormData.gender !== '' || 
      newFormData.location !== '';
    
    // Chỉ gửi dữ liệu nếu có thông tin được nhập
    onDataChange(hasData ? newFormData : null);
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3, borderRadius: 2, height: '100%' }}>
      <Typography variant="h5" sx={{ mb: 3, textAlign: 'center', color: 'primary.main' }}>
        Thông tin bệnh nhân (tùy chọn)
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Thông tin bệnh nhân sẽ giúp cải thiện độ chính xác trong chẩn đoán. 
        Bạn có thể để trống nếu không muốn cung cấp.
      </Typography>
      
      <Box>
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
            {/* Cập nhật giá trị phù hợp với model: male, female, unknown */}
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
    </Paper>
  );
};

export default PatientDataForm;