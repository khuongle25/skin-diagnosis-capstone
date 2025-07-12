import React, { useState } from "react";
import CustomSnackbar from "./CustomSnackbar";
import { useNavigate } from "react-router-dom";
import {
  Avatar,
  Button,
  CssBaseline,
  TextField,
  Grid,
  Link,
  Typography,
  Container,
} from "@mui/material";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import { useTheme } from "@mui/material";

// Import service xác thực
import { authService } from "../services/authService";

export default function SignUp() {
  const theme = useTheme();
  const [formData, setFormData] = useState({
    display_name: "",
    email: "",
    password: "",
  });
  const navigate = useNavigate();
  const [isSnackbarOpen, setIsSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarStatus, setSnackbarStatus] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Gọi API đăng ký từ backend
      const response = await authService.register({
        email: formData.email,
        password: formData.password,
        display_name: formData.display_name
      });
      
      // Nếu đăng ký thành công, tự động đăng nhập
      const loginResponse = await authService.login({
        email: formData.email,
        password: formData.password
      });
      
      // Lưu token và thông tin người dùng
      localStorage.setItem("access_token", loginResponse.id_token);
      localStorage.setItem("user", JSON.stringify(loginResponse.user));
      
      setSnackbarMessage("Tài khoản đã được tạo thành công!");
      setSnackbarStatus("success");
      setIsSnackbarOpen(true);
      
      // Sau 2 giây, chuyển hướng đến trang chủ
      setTimeout(() => {
        navigate("/home");
      }, 2000);
    } catch (error) {
      console.error("Error:", error);
      setSnackbarMessage(error.message || "Không thể đăng ký! Vui lòng thử lại.");
      setSnackbarStatus("error");
      setIsSnackbarOpen(true);
    }
  };

  return (
    <Container component="main" maxWidth="xs">
      <CssBaseline />
      <div
        style={{
          marginTop: theme.spacing(8),
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Avatar
          style={{
            margin: theme.spacing(1),
            backgroundColor: theme.palette.secondary.main,
          }}
        >
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Đăng ký
        </Typography>
        <form
          style={{
            width: "100%",
            marginTop: theme.spacing(3),
          }}
          onSubmit={handleSubmit}
        >
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                autoComplete="fname"
                name="display_name"
                variant="standard"
                required
                fullWidth
                id="fullName"
                label="Họ và tên"
                autoFocus
                value={formData.display_name}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                variant="standard"
                required
                fullWidth
                id="email"
                label="Địa chỉ Email"
                name="email"
                autoComplete="email"
                value={formData.email}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                variant="standard"
                required
                fullWidth
                name="password"
                label="Mật khẩu"
                type="password"
                id="password"
                autoComplete="current-password"
                value={formData.password}
                onChange={handleChange}
              />
            </Grid>
          </Grid>
          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            style={{ margin: theme.spacing(3, 0, 2) }}
          >
            Đăng ký
          </Button>

          <Grid container justifyContent="flex-end">
            <Grid item>
              <Link href="/signin" variant="body2">
                Đã có tài khoản? Đăng nhập
              </Link>
            </Grid>
          </Grid>
        </form>
        <CustomSnackbar
          open={isSnackbarOpen}
          message={snackbarMessage}
          handleClose={() => setIsSnackbarOpen(false)}
          status={snackbarStatus}
        />
      </div>
    </Container>
  );
}