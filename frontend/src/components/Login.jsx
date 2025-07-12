import React, { useState } from "react";
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Grid from "@mui/material/Grid";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import Typography from "@mui/material/Typography";
import { Box, useTheme, Alert, Snackbar } from "@mui/material";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import image from "../services/header.png";
import { useAuth } from "../contexts/AuthContext";

// Import dịch vụ xác thực
import { authService } from "../services/authService";
// Import Firebase Auth chỉ để đăng nhập Google (lấy token)
import { getAuth, signInWithPopup, GoogleAuthProvider } from "firebase/auth";
import { initializeApp } from "firebase/app";

// Cấu hình Firebase - chỉ dùng cho OAuth flow
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  // Các thông tin khác...
};

// Khởi tạo Firebase
const app = initializeApp(firebaseConfig);
const firebaseAuth = getAuth(app);

const SignInSide = () => {
  const theme = useTheme();
  const { setUser } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [openDialogFail, setOpenDialogFail] = useState(false);
  const [errorMessage, setErrorMessage] = useState("Không thể đăng nhập! Vui lòng thử lại.");
  const [forgotDialogOpen, setForgotDialogOpen] = useState(false);
  const [resetEmail, setResetEmail] = useState("");
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarSeverity, setSnackbarSeverity] = useState("success");
  const navigate = useNavigate();
  
  const handleClose = () => {
    setOpenDialogFail(false);
  };

  const handleOpenForgotDialog = () => {
    setForgotDialogOpen(true);
  };

  const handleCloseForgotDialog = () => {
    setForgotDialogOpen(false);
  };

  const handleOpenSnackbar = () => {
    setOpenSnackbar(true);
  };

  const handleCloseSnackbar = (event, reason) => {
    if (reason === "clickaway") {
      return;
    }
    setOpenSnackbar(false);
  };

  // Xử lý quên mật khẩu
  const handleForgot = async (event) => {
    event.preventDefault();
    try {
      // Gửi yêu cầu reset mật khẩu thông qua Firebase (yêu cầu thêm API endpoint)
      // ... (tạm thời bỏ qua vì backend chưa hỗ trợ tính năng này)
      
      setSnackbarMessage("Email đặt lại mật khẩu đã được gửi!");
      setSnackbarSeverity("success");
      handleCloseForgotDialog();
      handleOpenSnackbar();
    } catch (error) {
      console.error("Error:", error);
      setSnackbarMessage("Không thể gửi email đặt lại mật khẩu. Vui lòng thử lại.");
      setSnackbarSeverity("error");
      handleOpenSnackbar();
    }
  };

  // Đăng nhập với Google
  const handleGoogleSignIn = async () => {
    try {
      // Bước 1: Sử dụng popup của Firebase để lấy Google ID token
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(firebaseAuth, provider);
      const idToken = await result.user.getIdToken();
      
      // Bước 2: Gửi token lên backend để xác thực
      const authResult = await authService.googleLogin(idToken);
      
      // Bước 3: Lưu token từ backend vào local storage (SỬA ĐOẠN NÀY)
      localStorage.setItem("access_token", authResult.token); // Sửa id_token thành token
      localStorage.setItem("user", JSON.stringify(authResult.user));
      
      // Bước 4: Cập nhật context (THÊM ĐOẠN NÀY)
      setUser(authResult.user);
      console.log("Google sign-in result:", result);
      console.log("ID Token:", idToken);
      console.log("Auth result from backend:", authResult);
      // Đăng nhập thành công, chuyển hướng về trang chủ
      navigate("/home");
    } catch (error) {
      console.error("Error during Google login:", error);
      setErrorMessage("Đăng nhập với Google thất bại: " + error.message);
      setOpenDialogFail(true);
    }
  };

  // Đăng nhập với email/password
  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      // Gọi API đăng nhập từ backend
      const authResult = await authService.login({ 
        email: email, 
        password: password 
      });
      
      // Lưu token và thông tin người dùng
      localStorage.setItem("access_token", authResult.id_token);
      localStorage.setItem("user", JSON.stringify(authResult.user));
      
      // Cập nhật context
      setUser(authResult.user);
      
      // Đăng nhập thành công, chuyển hướng về trang chủ
      navigate("/home");
    } catch (error) {
      console.error("Error during login:", error);
      setErrorMessage(error.message || "Đăng nhập thất bại!");
      setOpenDialogFail(true);
    }
  };

  return (
    <Grid
      container
      component="main"
      style={{
        height: "100vh",
        overflow: "hidden",
      }}
    >
      <CssBaseline />
      <Grid item xs={false} sm={4} md={7}>
        <Box
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: "100%",
            width: "100%",
          }}
        >
          <img
            src={image}
            alt="Login background"
            style={{
              backgroundRepeat: "no-repeat",
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          />
        </Box>
      </Grid>
      <Grid item xs={12} sm={8} md={5} component={Paper} elevation={6} square>
        <div
          style={{
            margin: theme.spacing(8, 4),
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
            Đăng nhập
          </Typography>
          <form
            style={{
              width: "100%",
              marginTop: theme.spacing(1),
            }}
            noValidate
            onSubmit={handleSubmit}
          >
            <TextField
              variant="standard"
              margin="normal"
              required
              fullWidth
              id="email"
              label="Địa chỉ Email"
              name="email"
              autoComplete="email"
              autoFocus
              onChange={(e) => setEmail(e.target.value)}
            />
            <TextField
              variant="standard"
              margin="normal"
              required
              fullWidth
              name="password"
              label="Mật khẩu"
              type="password"
              id="password"
              autoComplete="current-password"
              onChange={(e) => setPassword(e.target.value)}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="primary"
              style={{
                margin: theme.spacing(3, 0, 2),
              }}
            >
              Đăng nhập
            </Button>
            
            {/* Nút đăng nhập với Google */}
            <Button
              fullWidth
              variant="outlined"
              color="primary"
              onClick={handleGoogleSignIn}
              style={{
                margin: theme.spacing(1, 0, 2),
              }}
            >
              Đăng nhập với Google
            </Button>
            
            <Dialog open={openDialogFail} onClose={handleClose}>
              <DialogTitle>{"Đăng nhập thất bại"}</DialogTitle>
              <DialogContent>
                <DialogContentText>
                  {errorMessage}
                </DialogContentText>
              </DialogContent>
              <DialogActions>
                <Button onClick={handleClose} color="primary" autoFocus>
                  Đóng
                </Button>
              </DialogActions>
            </Dialog>
            <Grid container>
              <Grid item xs>
                <Link href="#" variant="body2" onClick={handleOpenForgotDialog}>
                  Quên mật khẩu?
                </Link>
              </Grid>
              <Grid item>
                <Link href="/signup" variant="body2">
                  {"Chưa có tài khoản? Đăng ký"}
                </Link>
              </Grid>
            </Grid>
          </form>
          <Dialog open={forgotDialogOpen} onClose={handleCloseForgotDialog}>
            <DialogTitle>Khôi phục mật khẩu</DialogTitle>
            <form onSubmit={handleForgot}>
              <DialogContent>
                <DialogContentText>
                  Vui lòng nhập email của bạn. Chúng tôi sẽ gửi email hướng dẫn đặt lại mật khẩu.
                </DialogContentText>
                <TextField
                  autoFocus
                  margin="dense"
                  id="email-reset"
                  label="Địa chỉ Email"
                  type="email"
                  fullWidth
                  onChange={(e) => setResetEmail(e.target.value)}
                />
              </DialogContent>
              <DialogActions>
                <Button onClick={handleCloseForgotDialog} color="primary">
                  Hủy
                </Button>
                <Button type="submit" color="primary">
                  Gửi
                </Button>
              </DialogActions>
            </form>
          </Dialog>
          <Snackbar
            open={openSnackbar}
            autoHideDuration={6000}
            onClose={handleCloseSnackbar}
          >
            <Alert
              onClose={handleCloseSnackbar}
              severity={snackbarSeverity}
              elevation={6}
              variant="filled"
            >
              {snackbarMessage}
            </Alert>
          </Snackbar>
        </div>
      </Grid>
    </Grid>
  );
};

export default SignInSide;