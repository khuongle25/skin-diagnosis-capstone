import axios from 'axios';
import { authHeader } from '../utils/auth';
import { getCodeFromFullName } from './LesionDescriptions';

// Lấy URL API từ biến môi trường hoặc sử dụng mặc định
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const sendChatMessage = async (message, lesionType, userId, conversationId, enhanceKnowledge = false) => {
  try {
    console.log("Đang gửi tin nhắn đến API:", message, lesionType);
    
    if (!conversationId) {
      throw new Error("Cần khởi tạo chatbot trước khi gửi tin nhắn!");
    }
    const lesionCode = getCodeFromFullName(lesionType);
    const response = await axios.post(`${API_URL}/api/chat`, {
      message: message,
      lesion_type: lesionCode,
      user_id: userId,
      conversation_id: conversationId,
      enhance_knowledge: enhanceKnowledge
    }, {
      headers: authHeader()
    });
    
    return {
      response: response.data.response,
      conversationId: response.data.conversation_id,
      lesionType: response.data.lesion_type
    };
  } catch (error) {
    console.error("Lỗi khi gọi API chat:", error);
    throw new Error("Không thể kết nối với dịch vụ chat. Vui lòng thử lại sau.");
  }
};

export const getUserConversations = async (userId) => {
  try {
    console.log("Lấy lịch sử hội thoại cho user:", userId);
    
    const response = await axios.get(`${API_URL}/api/conversations/${userId}`, {
      headers: authHeader()
    });
    
    // Kiểm tra cấu trúc của response.data
    console.log("Response data:", response.data);
    
    // Trả về mảng các cuộc hội thoại
    // Nếu response.data là một mảng, trả về nó
    // Nếu response.data.conversations tồn tại, trả về nó
    // Nếu không, trả về mảng rỗng
    if (Array.isArray(response.data)) {
      return response.data;
    } else if (response.data && Array.isArray(response.data.conversations)) {
      return response.data.conversations;
    } else {
      console.error("Cấu trúc dữ liệu không đúng:", response.data);
      return [];
    }
  } catch (error) {
    console.error("Lỗi khi lấy lịch sử hội thoại:", error);
    throw error;
  }
};

// Thêm hàm initializeChatbot
export const initializeChatbot = async (lesionType, userId) => {
  try {
    console.log("Khởi tạo chatbot cho loại tổn thương:", lesionType);
    const lesionCode = getCodeFromFullName(lesionType);
    const response = await axios.post(`${API_URL}/api/initialize-chatbot`, {
      lesion_type: lesionCode,
      user_id: userId
    }, {
      headers: authHeader()
    });
    
    return {
      conversationId: response.data.conversation_id,
      lesionType: response.data.lesion_type,
      success: response.data.success
    };
  } catch (error) {
    console.error("Lỗi khi khởi tạo chatbot:", error);
    throw new Error("Không thể khởi tạo chatbot. Vui lòng thử lại sau.");
  }
};

export const getConversationHistory = async (conversationId) => {
  try {
    console.log("Fetching conversation history:", conversationId);
    
    const response = await axios.get(`${API_URL}/api/conversation/${conversationId}`, {
      headers: authHeader()
    });
    
    return response.data;
  } catch (error) {
    console.error("Lỗi khi lấy lịch sử trò chuyện:", error);
    throw new Error("Không thể lấy lịch sử trò chuyện. Vui lòng thử lại sau.");
  }
};