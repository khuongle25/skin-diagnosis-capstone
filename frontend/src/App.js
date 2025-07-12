// src/App.jsx
import React, { useState } from "react";
import {
  Container,
  Typography,
  Box,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Grid,
  Alert,
} from "@mui/material";
import axios from "axios";

// Components
import { IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import ImageUpload from "./components/ImageUpload";
import ImageCropper from "./components/ImageCropper";
import ResultDisplay from "./components/ResultDisplay";
import PatientDataForm from "./components/PatientDataForm";
import AlertDialog from "./components/AlertDialog";
import ImagePreview from "./components/ImagePreview";
import NavBar from "./components/NavBar";
import DiagnosisLoading from "./components/DiagnosisLoading";
import ChatInterface from "./components/ChatInterface";
import ChatHistoryPage from "./components/ChatHistoryPage";
import ImageAnalysisLoading from "./components/ImageAnalysisLoading";
import CommunityFeed from "./components/CommunityFeed";
import UserProfile from "./components/UserProfile";
import { Snackbar } from "@mui/material";

import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./components/Login";
import Register from "./components/Register";
import { useAuth } from "./contexts/AuthContext";
import { initializeChatbot } from "./services/chatService";
import {
  getLesionDescription,
  getCodeFromFullName,
} from "./services/LesionDescriptions";

// API configuration
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Theme
const theme = createTheme({
  palette: {
    primary: {
      main: "#3f51b5",
    },
    secondary: {
      main: "#f50057",
    },
  },
});

// Sửa PrivateRoute trong App.js
function PrivateRoute({ children }) {
  const { user } = useAuth(); // Thay vì currentUser

  if (!user) {
    return <Navigate to="/signin" />;
  }

  return children;
}

function App() {
  const [file, setFile] = useState(null);
  const [fileSrc, setFileSrc] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showCropper, setShowCropper] = useState(false);
  const [alertOpen, setAlertOpen] = useState(false);
  const [alertMessage, setAlertMessage] = useState("");
  const [showImagePreview, setShowImagePreview] = useState(false);
  const [patientData, setPatientData] = useState(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const { user } = useAuth();
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [chatLesionType, setChatLesionType] = useState(null);
  const [visualizeDone, setVisualizeDone] = useState(false);
  const [analyzingImage, setAnalyzingImage] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");

  const handleSnackbarClose = (event, reason) => {
    if (reason === "clickaway") return;
    setSnackbarOpen(false);
  };

  const handleImageSelected = (selectedFile) => {
    setFile(selectedFile);
    setFileSrc(URL.createObjectURL(selectedFile));
    checkIfSkinImage(selectedFile);
  };

  const handleCropComplete = async (croppedFile) => {
    setFile(croppedFile);
    setFileSrc(URL.createObjectURL(croppedFile));
    setShowCropper(false);
    setShowImagePreview(true); // Sau khi crop mới cho xem preview
  };

  const handleCropCancel = () => {
    setShowCropper(false);
    setFile(null);
    setFileSrc(null);
  };

  const handleCloseAlert = () => {
    setAlertOpen(false);
    setFile(null);
    setFileSrc(null);
  };

  const handleOpenCropper = () => {
    setAlertOpen(false);
    setShowCropper(true);
  };

  const handlePatientDataChange = (data) => {
    // Kiểm tra xem data có rỗng không
    const isEmpty =
      !data ||
      ((!data.age || data.age === "") &&
        (!data.gender || data.gender === "") &&
        (!data.location || data.location === ""));

    // Nếu tất cả đều trống, đặt patientData = null
    setPatientData(isEmpty ? null : data);
  };

  const handleDiagnose = () => {
    if (file) {
      setDiagnosing(true); // Bắt đầu hiển thị màn hình chẩn đoán
      setShowImagePreview(false);
      processImageWithMetadata(file, patientData);
    }
  };

  const handleCancelDiagnosis = () => {
    setFile(null);
    setFileSrc(null);
    setShowImagePreview(false);
  };

  const handleReset = () => {
    setFile(null);
    setFileSrc(null);
    setError(null);
    setResult(null);
    setShowCropper(false);
    setShowImagePreview(false);
    setPatientData(null);
    setDiagnosing(false);
    setShowChatbot(false);

    // Reset conversationId khi bắt đầu phân tích ảnh mới
    setCurrentConversationId(null);
  };

  const handleUpdateMetadata = (updatedData) => {
    setPatientData(updatedData);
  };

  const handleOpenChatbot = async (lesionType) => {
    setLoading(true);
    try {
      if (!currentConversationId) {
        // Cần chuyển về mã để gửi cho backend
        const lesionCode = getCodeFromFullName(lesionType);
        const chatbotData = await initializeChatbot(lesionCode, user?.uid);
        setCurrentConversationId(chatbotData.conversationId);
        setChatLesionType(lesionType); // Lưu tên đầy đủ cho frontend
      }
      // Hiển thị chatbot
      setShowChatbot(true);
    } catch (error) {
      console.error("Lỗi khởi tạo chatbot:", error);
      alert("Không thể khởi tạo chatbot tư vấn. Vui lòng thử lại sau.");
    } finally {
      setLoading(false);
    }
  };

  const handleToggleChatbot = () => {
    setShowChatbot(!showChatbot);
  };

  const handleCloseChatbot = () => {
    setShowChatbot(false);
  };

  const checkIfSkinImage = async (imageFile) => {
    const formData = new FormData();
    formData.append("file", imageFile);

    setLoading(true);
    setAnalyzingImage(true); // Bắt đầu hiển thị loading
    setError(null); // Reset error state

    try {
      const classifyResponse = await axios.post(
        `${API_URL}/analyze/`,
        formData
      );
      if (classifyResponse.data.is_skin) {
        // Nếu là ảnh da, luôn yêu cầu crop lại thành vuông
        setShowCropper(true); // <-- luôn crop lại
        // setShowImagePreview(true); // bỏ dòng này
      } else {
        // Nếu không phải ảnh da, hiển thị hộp thoại cảnh báo
        setAlertMessage(
          classifyResponse.data.message ||
            "Ảnh không phải là ảnh da người. Vui lòng tải ảnh khác hoặc crop ảnh này."
        );
        setAlertOpen(true);

        // Nếu là ảnh đã crop mà vẫn không phải ảnh da, thêm thông báo
        // if (isCroppedImage) {
        //   setAlertMessage(
        //     "Ảnh đã crop vẫn không được nhận dạng là ảnh da người. Vui lòng crop lại hoặc chọn ảnh khác."
        //   );
        // }
      }
    } catch (err) {
      console.error("Error processing image:", err);
      setError(
        err.response?.data?.detail || "Lỗi khi xử lý ảnh. Vui lòng thử lại."
      );
      setFile(null);
      setFileSrc(null);
    } finally {
      setAnalyzingImage(false);
    }
  };

  const processImageWithMetadata = async (imageFile, metadata) => {
    const formData = new FormData();
    formData.append("file", imageFile);

    // Log để kiểm tra
    console.log("Sending metadata to API:", metadata);

    // Chỉ gửi metadata nếu có dữ liệu
    const hasMetadata =
      metadata &&
      ((metadata.age && metadata.age !== "") ||
        (metadata.gender && metadata.gender !== "") ||
        (metadata.location && metadata.location !== ""));

    if (hasMetadata) {
      const cleanedMetadata = {
        age: metadata.age || null,
        gender: metadata.gender || null,
        location: metadata.location || null,
      };
      formData.append("metadata", JSON.stringify(cleanedMetadata));
    }

    setLoading(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 4000));

      const segmentResponse = await axios.post(
        `${API_URL}/classify/`,
        formData
      );
      const resultWithMetadata = {
        ...segmentResponse.data,
        metadata: hasMetadata ? metadata : null,
      };

      setResult(resultWithMetadata);
    } catch (err) {
      console.error("Error processing image:", err);
      // Kiểm tra lỗi phân đoạn không ra tổn thương
      const detail = err.response?.data?.detail;
      if (
        detail &&
        detail.includes(
          "Không có tổn thương nào được mô hình phân đoạn tìm thấy"
        )
      ) {
        setSnackbarMessage(
          "Không có tổn thương nào được mô hình phân đoạn tìm thấy."
        );
        setSnackbarOpen(true);
        setFile(null);
        setFileSrc(null);
        setShowImagePreview(false);
        setDiagnosing(false);
        setResult(null);
        return;
      }
      setError(detail || "Lỗi khi xử lý ảnh. Vui lòng thử lại.");
    } finally {
      setLoading(false);
      setDiagnosing(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ImageAnalysisLoading open={analyzingImage} />
      <Routes>
        <Route path="/signin" element={<Login />} />
        <Route path="/signup" element={<Register />} />
        <Route
          path="/chat-history"
          element={
            <PrivateRoute>
              <ChatHistoryPage />
            </PrivateRoute>
          }
        />
        <Route
          path="/community"
          element={
            <PrivateRoute>
              <NavBar />
              <CommunityFeed />
            </PrivateRoute>
          }
        />
        <Route
          path="/my-posts"
          element={
            <PrivateRoute>
              <NavBar />
              <UserProfile />
            </PrivateRoute>
          }
        />
        <Route
          path="/home"
          element={
            <PrivateRoute>
              <NavBar />
              <Container maxWidth="lg">
                <Box sx={{ my: 4 }}>
                  <Typography
                    variant="h3"
                    component="h1"
                    align="center"
                    gutterBottom
                    color="primary.main"
                  >
                    CÔNG CỤ PHÂN TÍCH TỔN THƯƠNG DA LIỄU
                  </Typography>
                  <Typography
                    variant="h6"
                    align="center"
                    color="text.secondary"
                    paragraph
                  >
                    Tải lên ảnh da để phát hiện và phân đoạn tổn thương
                  </Typography>

                  {error && (
                    <Alert severity="error" sx={{ mb: 3 }}>
                      {error}
                    </Alert>
                  )}

                  {/* Dialog thông báo không phải ảnh da */}
                  <AlertDialog
                    open={alertOpen}
                    message={alertMessage}
                    onClose={handleCloseAlert}
                    onCrop={handleOpenCropper}
                  />

                  {/* Hiển thị form tải ảnh và thông tin bệnh nhân ban đầu */}
                  {!file &&
                    !showCropper &&
                    !showImagePreview &&
                    !diagnosing &&
                    !result && (
                      <Grid container spacing={3} sx={{ mt: 2 }}>
                        <Grid item xs={12} md={7}>
                          <ImageUpload onImageSelected={handleImageSelected} />
                        </Grid>
                        <Grid item xs={12} md={5}>
                          <PatientDataForm
                            onDataChange={handlePatientDataChange}
                          />
                        </Grid>
                      </Grid>
                    )}

                  {/* Hiển thị công cụ cắt ảnh nếu cần */}
                  {showCropper && (
                    <Box sx={{ my: 4 }}>
                      <ImageCropper
                        imageSrc={fileSrc}
                        onCropComplete={handleCropComplete}
                        onCancel={handleCropCancel}
                      />
                    </Box>
                  )}

                  {/* Hiển thị màn hình xem trước ảnh và xác nhận chẩn đoán */}
                  {showImagePreview && (
                    <ImagePreview
                      imageSrc={fileSrc}
                      onDiagnose={handleDiagnose}
                      onCancel={handleCancelDiagnosis}
                      patientData={patientData}
                      onUpdateMetadata={handleUpdateMetadata}
                    />
                  )}

                  {/* Hiển thị màn hình đang chẩn đoán */}
                  {diagnosing && (
                    <DiagnosisLoading hasMetadata={!!patientData} />
                  )}

                  {/* Hiển thị kết quả chẩn đoán */}
                  {result && !diagnosing && (
                    <ResultDisplay
                      result={result}
                      loading={loading}
                      onReset={handleReset}
                      showChatbot={showChatbot}
                      onOpenChatbot={handleOpenChatbot}
                      onCloseChatbot={handleCloseChatbot}
                      currentConversationId={currentConversationId}
                      onVisualizeDone={() => setVisualizeDone(true)}
                    />
                  )}

                  {showChatbot && !result && !diagnosing && visualizeDone && (
                    <Box
                      sx={{ mt: 3, display: "flex", justifyContent: "center" }}
                    >
                      <Box
                        sx={{
                          width: "100%",
                          maxWidth: "600px",
                          position: "relative",
                        }}
                      >
                        <Box sx={{ position: "absolute", top: 0, right: 0 }}>
                          <IconButton onClick={handleCloseChatbot} size="small">
                            <CloseIcon />
                          </IconButton>
                        </Box>
                        <ChatInterface
                          lesionType={chatLesionType}
                          conversationId={currentConversationId}
                        />
                      </Box>
                    </Box>
                  )}
                </Box>
                <Snackbar
                  open={snackbarOpen}
                  autoHideDuration={5000}
                  onClose={handleSnackbarClose}
                  anchorOrigin={{ vertical: "top", horizontal: "right" }}
                >
                  <Alert
                    onClose={handleSnackbarClose}
                    severity="error"
                    sx={{
                      width: "100%",
                      bgcolor: "#d32f2f", // đỏ đậm hơn mặc định
                      color: "#fff",
                      fontWeight: "bold",
                      fontSize: "1.2rem",
                      boxShadow: 6,
                      borderRadius: 3,
                      py: 2,
                      px: 3,
                      alignItems: "center",
                      letterSpacing: 1,
                    }}
                  >
                    {snackbarMessage}
                  </Alert>
                </Snackbar>
              </Container>
            </PrivateRoute>
          }
        />
        <Route path="/" element={<Navigate to="/home" />} />
      </Routes>
    </ThemeProvider>
  );
}

export default App;
