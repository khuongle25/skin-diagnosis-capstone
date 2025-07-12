// src/components/AlertDialog.jsx - Thêm chỉ báo cho ảnh đã crop
import React from 'react';
import { 
  Dialog, DialogActions, DialogContent, DialogContentText, 
  DialogTitle, Button, Typography 
} from '@mui/material';
import ErrorIcon from '@mui/icons-material/Error';
import ContentCutIcon from '@mui/icons-material/ContentCut';

const AlertDialog = ({ open, message, onClose, onCrop, isCroppedImage = false }) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ErrorIcon color="error" />
        {"Không phải ảnh da người"}
      </DialogTitle>
      <DialogContent>
        <DialogContentText id="alert-dialog-description">
          {message}
        </DialogContentText>
        
        {isCroppedImage && (
          <Typography variant="body2" sx={{ mt: 2, color: 'warning.main', display: 'flex', alignItems: 'center', gap: 1 }}>
            <ContentCutIcon fontSize="small" />
            Lưu ý: Ảnh đã được crop nhưng vẫn không được nhận diện là ảnh da người.
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Tải ảnh khác
        </Button>
        <Button onClick={onCrop} color="secondary" autoFocus>
          {isCroppedImage ? 'Crop lại ảnh này' : 'Crop ảnh này'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AlertDialog;