// src/components/ImageCropper.jsx
import React, { useState, useRef } from 'react';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { Box, Button, Typography, Paper } from '@mui/material';

function centerAspectCrop(mediaWidth, mediaHeight, aspect) {
  return centerCrop(
    makeAspectCrop(
      {
        unit: '%',
        width: 90,
      },
      aspect,
      mediaWidth,
      mediaHeight,
    ),
    mediaWidth,
    mediaHeight,
  );
}

const ImageCropper = ({ imageSrc, onCropComplete, onCancel }) => {
  const [crop, setCrop] = useState();
  const [completedCrop, setCompletedCrop] = useState(null);
  const imageRef = useRef(null);

  function onImageLoad(e) {
    const { width, height } = e.currentTarget;
    const aspect = 1;
    setCrop(centerAspectCrop(width, height, aspect));
  }

  const handleCropComplete = () => {
    if (!completedCrop || !imageRef.current) return;

    const image = imageRef.current;
    const canvas = document.createElement('canvas');
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;
    
    canvas.width = completedCrop.width;
    canvas.height = completedCrop.height;
    
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(
      image,
      completedCrop.x * scaleX,
      completedCrop.y * scaleY,
      completedCrop.width * scaleX,
      completedCrop.height * scaleY,
      0,
      0,
      completedCrop.width,
      completedCrop.height,
    );
    
    // Convert canvas to blob and create a file
    canvas.toBlob((blob) => {
      if (!blob) return;
      const croppedImage = new File([blob], 'cropped-image.jpg', { type: 'image/jpeg' });
      onCropComplete(croppedImage);
    }, 'image/jpeg');
  };

  return (
    <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
      <Typography variant="h5" sx={{ mb: 2, color: 'primary.main', textAlign: 'center' }}>
        Chọn vùng tổn thương cụ thể
      </Typography>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <ReactCrop
          crop={crop}
          onChange={c => setCrop(c)}
          onComplete={c => setCompletedCrop(c)}
          aspect={1}
        >
          <img 
            ref={imageRef}
            src={imageSrc}
            onLoad={onImageLoad}
            alt="Crop me"
            style={{ maxHeight: '500px', maxWidth: '100%' }}
          />
        </ReactCrop>
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button 
          variant="outlined" 
          color="secondary" 
          onClick={onCancel}
        >
          Cancel
        </Button>
        <Button 
          variant="contained" 
          color="primary"
          onClick={handleCropComplete}
          disabled={!completedCrop?.width || !completedCrop?.height}
        >
          Apply Crop
        </Button>
      </Box>
    </Paper>
  );
};

export default ImageCropper;