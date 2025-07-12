import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../services/authService';
import { getToken, clearSession, getUser } from '../utils/auth';

// Tạo context
const AuthContext = createContext(null);

// Hook để sử dụng context
export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Kiểm tra trạng thái đăng nhập khi component được mount
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        // Thử lấy thông tin người dùng từ localStorage trước
        const userFromStorage = getUser();
        const token = getToken();
        
        if (userFromStorage && token) {
          setUser(userFromStorage);
          // Nếu có thể, xác thực token với backend
          try {
            const userInfo = await authService.getUserProfile(token);
            setUser(userInfo);
          } catch (err) {
            console.log("Token validation failed, using cached user data");
          }
        }
      } catch (error) {
        console.error('Auth verification failed:', error);
        clearSession();
      } finally {
        setLoading(false);
      }
    };

    checkAuthStatus();
  }, []);

  // Hàm đăng xuất
  const logout = () => {
    clearSession();
    setUser(null);
    window.location.href = '/signin';
  };

  // Các giá trị được chia sẻ qua context
  const value = {
    user,
    loading,
    error,
    setUser,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {loading ? <div>Đang tải...</div> : children}
    </AuthContext.Provider>
  );
};