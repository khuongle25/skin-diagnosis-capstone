// src/components/ImageUpload.jsx
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Paper } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const ImageUpload = ({ onImageSelected }) => {
  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      onImageSelected(file);
    }
  }, [onImageSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1
  });

  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        backgroundColor: theme => isDragActive ? theme.palette.primary.light : theme.palette.background.paper,
        borderRadius: 2,
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'background-color 0.3s',
        '&:hover': {
          backgroundColor: theme => theme.palette.primary.light,
        }
      }}
      {...getRootProps()}
    >
      <input {...getInputProps()} />
      <Box sx={{ py: 5, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main' }} />
        <Typography variant="h5" color="primary.main">
          {isDragActive ? 'Drop the image here' : 'Kéo và thả ảnh vào đây - Hoặc click để tải ảnh lên'}
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Chất lượng ảnh cao quyết định tính chính xác của chẩn đoán.
          <br />
          Hỗ trợ định dạng: JPG, JPEG, PNG
        </Typography>
      </Box>
    </Paper>
  );
};

export default ImageUpload;