import React, { useState } from "react";
import {
    Box,
    Typography,
    CircularProgress,
    Backdrop,
} from "@mui/material";

const ImageAnalysisLoading = ({ open }) => (
    <Backdrop
      sx={{
        color: "#fff",
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backdropFilter: "blur(3px)",
        backgroundColor: "rgba(0, 0, 0, 0.7)",
      }}
      open={open}
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          p: 4,
          bgcolor: "background.paper",
          borderRadius: 2,
          color: "text.primary",
          boxShadow: 24,
        }}
      >
        <CircularProgress size={60} sx={{ mb: 2 }} />
        <Typography variant="h6" sx={{ mb: 1 }}>
          Đang phân tích ảnh...
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center">
          Đang kiểm tra xem đây có phải là ảnh da không
        </Typography>
      </Box>
    </Backdrop>
  );

  export default ImageAnalysisLoading;