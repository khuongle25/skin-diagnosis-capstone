/**
 * Các tiện ích xác thực
 */

// Lấy token từ localStorage
export const getToken = () => {
    return localStorage.getItem('access_token');
  };
  
  // Kiểm tra xem người dùng đã đăng nhập chưa
  export const isAuthenticated = () => {
    return !!getToken();
  };
  
  // Lấy thông tin người dùng từ localStorage
  export const getUser = () => {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
  };
  
  // Lưu thông tin phiên đăng nhập
  export const setSession = (token, user) => {
    localStorage.setItem('access_token', token);
    localStorage.setItem('user', JSON.stringify(user));
  };
  
  // Xóa phiên đăng nhập
  export const clearSession = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
  };
  
  // Lấy headers xác thực
  export const authHeader = () => {
    const token = getToken();
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  };