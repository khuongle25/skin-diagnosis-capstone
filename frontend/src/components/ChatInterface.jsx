import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  CircularProgress, 
  Typography, 
  IconButton, 
  FormControlLabel,
  Switch,
  Menu,
  MenuItem,
  Divider,
  Tooltip
} from '@mui/material';
import PublicIcon from '@mui/icons-material/Public'; // Thêm icon global
import LanguageIcon from '@mui/icons-material/Language'; // Hoặc icon này cũng được
import SendIcon from '@mui/icons-material/Send';
import HistoryIcon from '@mui/icons-material/History';
import SearchIcon from '@mui/icons-material/Search';
import { initializeChatbot } from '../services/chatService';
import { sendChatMessage, getUserConversations, getConversationHistory } from '../services/chatService';
import ChatMessage from './ChatMessage';
import { useAuth } from '../contexts/AuthContext';
import { format } from 'date-fns';
import viLocale from 'date-fns/locale/vi';
import CrawlVisualizationPopup from './CrawlVisualizationPopup';

const ChatInterface = ({ lesionType, conversationId: initialConversationId }) => {  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [enhanceKnowledge, setEnhanceKnowledge] = useState(false);
  const [historyAnchorEl, setHistoryAnchorEl] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [showCrawlVisualization, setShowCrawlVisualization] = useState(false);
  const [popupMode, setPopupMode] = useState('enhance');
  const [showInitVisualizer, setShowInitVisualizer] = useState(false);
  
  const messagesEndRef = useRef(null);
  const { user } = useAuth();

  // Tự động cuộn xuống tin nhắn mới nhất
  const scrollToBottom = () => {
    // Thêm timeout đảm bảo DOM đã cập nhật
    setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ 
          behavior: 'smooth',
          block: 'end'  // Đảm bảo cuộn tới cuối cùng
        });
      }
    }, 50);
  };

  useEffect(() => {
    if (initialConversationId && initialConversationId !== conversationId) {
      setConversationId(initialConversationId);
    }
  }, [initialConversationId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);


  // Thêm tin nhắn chào mừng khi component được tạo
  useEffect(() => {
    if (!conversationId) {
      setMessages([
        {
          id: 1,
          content: `Xin chào! Tôi có thể cung cấp thông tin về tổn thương da loại: ${lesionType}. Bạn muốn biết thêm điều gì?`,
          isUser: false,
        }
      ]);
    }
  }, [lesionType, conversationId]);

  useEffect(() => {
    const initializeChat = async () => {
      // CHỈ khởi tạo khi có lesionType và KHÔNG có conversationId
      // Thêm điều kiện để kiểm tra xem đây có phải là conversation từ lịch sử không
      if (lesionType && !conversationId && !initialConversationId) {
        try {
          // Đặt chế độ khởi tạo và hiển thị popup
          console.log("Initializing new chat for", lesionType);
          setPopupMode('initialization');
          setShowCrawlVisualization(true);
          
          // Đóng popup sau khi hoàn thành
          setTimeout(() => {
            setShowCrawlVisualization(false);
          }, 5000);
        } catch (error) {
          console.error("Error initializing chatbot:", error);
        }
      }
    };
    
    initializeChat();
  }, [lesionType, conversationId, initialConversationId]);
  
  // Thêm useEffect để tải lịch sử khi conversationId thay đổi
  useEffect(() => {
    const loadConversationHistory = async () => {
      if (conversationId) {
        try {
          setLoading(true);
          setMessages([]); // Xóa tin nhắn hiện tại trước khi tải mới
          
          const conversationData = await getConversationHistory(conversationId);
          
          if (conversationData.history && conversationData.history.length > 0) {
            // Chuyển đổi định dạng tin nhắn từ database sang định dạng UI
            const formattedMessages = conversationData.history.map(msg => ({
              id: Date.now() + Math.random(),
              content: msg.content,
              isUser: msg.role === 'human',
              timestamp: new Date().toISOString()
            }));
            
            setMessages(formattedMessages);
          } else {
            // Thêm case xử lý khi history trống
            setMessages([
              {
                id: Date.now(),
                content: `Xin chào! Tôi có thể cung cấp thông tin về tổn thương da loại: ${lesionType}. Bạn muốn biết thêm điều gì?`,
                isUser: false,
              }
            ]);
          }
          
          // Đảm bảo cuộn xuống
          setTimeout(() => {
            scrollToBottom();
          }, 100);
        } catch (error) {
          console.error("Lỗi tải lịch sử:", error);
          // Thêm thông báo lỗi
          setMessages([
            {
              id: Date.now(),
              content: "Xin lỗi, đã xảy ra lỗi khi tải cuộc hội thoại. Vui lòng thử lại.",
              isUser: false,
            }
          ]);
        } finally {
          setLoading(false);
        }
      }
    };
    
    loadConversationHistory();
  }, [conversationId, lesionType]); // Thêm lesionType vào dependencies

  // Tải lịch sử cuộc hội thoại
  const loadConversationHistory = async () => {
    if (!user?.uid) return;
    
    try {
      setLoadingHistory(true);
      const userConversations = await getUserConversations(user.uid);
      setConversations(userConversations);
      setLoadingHistory(false);
    } catch (error) {
      console.error("Lỗi khi tải lịch sử hội thoại:", error);
      setLoadingHistory(false);
    }
  };

  // Mở menu lịch sử
  const handleHistoryClick = (event) => {
    loadConversationHistory();
    setHistoryAnchorEl(event.currentTarget);
  };

  // Đóng menu lịch sử
  const handleHistoryClose = () => {
    setHistoryAnchorEl(null);
  };

  // Chọn một cuộc hội thoại cũ
  const handleSelectConversation = async (selectedConversationId) => {
    setConversationId(selectedConversationId);
    setHistoryAnchorEl(null);
    setMessages([]);
    setLoading(true);
    
    try {
      // Thay vì gửi tin nhắn, chỉ tải lịch sử hội thoại
      const conversationData = await getConversationHistory(selectedConversationId);
      
      if (conversationData.history && conversationData.history.length > 0) {
        // Chuyển đổi định dạng tin nhắn từ database sang định dạng UI
        const formattedMessages = conversationData.history.map(msg => ({
          id: Date.now() + Math.random(), // Tạo ID ngẫu nhiên
          content: msg.content,
          isUser: msg.role === 'human',
          timestamp: new Date().toISOString()
        }));
        
        setMessages(formattedMessages);
      } else {
        // Nếu không có lịch sử, hiển thị tin nhắn chào mừng
        setMessages([
          {
            id: Date.now(),
            content: `Xin chào! Tôi có thể cung cấp thông tin về tổn thương da loại: ${lesionType}. Bạn muốn biết thêm điều gì?`,
            isUser: false,
          }
        ]);
      }
    } catch (error) {
      console.error("Lỗi khi tải cuộc hội thoại:", error);
      setMessages([
        {
          id: Date.now(),
          content: "Xin lỗi, đã xảy ra lỗi khi tải cuộc hội thoại. Vui lòng thử lại.",
          isUser: false,
        }
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => {
        scrollToBottom();
      }, 100);
    }
  };

  // Bắt đầu cuộc hội thoại mới
  const handleNewConversation = async () => {
    try {
      setLoading(true); // Hiện loading
      setHistoryAnchorEl(null); // Đóng menu lịch sử
      setPopupMode('initialization');
      setShowCrawlVisualization(true);
      // Khởi tạo cuộc hội thoại mới thực sự bằng cách gọi API
      const chatbotData = await initializeChatbot(lesionType, user?.uid);
      
      // Cập nhật conversationId mới
      setConversationId(chatbotData.conversationId);
      
      // Reset messages với tin nhắn chào mừng
      setMessages([
        {
          id: Date.now(),
          content: `Xin chào! Tôi có thể cung cấp thông tin về tổn thương da loại: ${lesionType}. Bạn muốn biết thêm điều gì?`,
          isUser: false,
        }
      ]);
      
      console.log("Đã tạo cuộc hội thoại mới với ID:", chatbotData.conversationId);
    } catch (error) {
      console.error("Lỗi khi tạo cuộc hội thoại mới:", error);
      // Hiển thị thông báo lỗi cho người dùng
      setMessages([
        {
          id: Date.now(),
          content: "Xin lỗi, không thể tạo cuộc hội thoại mới. Vui lòng thử lại sau.",
          isUser: false,
        }
      ]);
      setTimeout(() => {
        setShowCrawlVisualization(false);
      }, 5000); // Đóng sau 5 giây
    } finally {
      setLoading(false); // Tắt loading
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    // Thêm tin nhắn của người dùng
    const userMessage = {
      id: Date.now(),
      content: inputValue,
      isUser: true,
    };
    
    setMessages([...messages, userMessage]);
    const userQuestion = inputValue;
    setInputValue('');
    setLoading(true);
    if (enhanceKnowledge) {
      console.log("Kích hoạt popup crawl");
      setPopupMode('enhance');
      setShowCrawlVisualization(true);
    }
    
    try {
      // Gọi API thực tế
      const chatResponse = await sendChatMessage(
        userQuestion, 
        lesionType, 
        user?.uid || 'anonymous', 
        conversationId, 
        enhanceKnowledge
      );
      
      // Cập nhật conversation_id nếu đây là cuộc hội thoại mới
      if (!conversationId && chatResponse.conversationId) {
        setConversationId(chatResponse.conversationId);
      }
      
      // Thêm phản hồi từ bot
      const botResponse = {
        id: Date.now() + 1,
        content: chatResponse.response,
        isUser: false,
      };
      
      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      console.error("Lỗi:", error);
      
      // Thêm thông báo lỗi
      const errorMessage = {
        id: Date.now() + 1,
        content: "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
        isUser: false,
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%',
      minHeight: '500px',
      width: '100%',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Header với tiêu đề và các nút điều khiển */}
      <Box sx={{ 
  p: 2, 
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  color: 'white',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  borderTopLeftRadius: 12,
  borderTopRightRadius: 12,
  flexShrink: 0,
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'linear-gradient(45deg, rgba(255, 255, 255, 0.1) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.1) 50%, rgba(255, 255, 255, 0.1) 75%, transparent 75%, transparent)',
    backgroundSize: '20px 20px',
    opacity: 0.1,
    pointerEvents: 'none'
  }
}}>
  {/* Phần bên trái với icon và tiêu đề */}
  <Box sx={{ 
    display: 'flex', 
    alignItems: 'center', 
    gap: 1.5,
    zIndex: 1 
  }}>
     
    {/* Tiêu đề với typography đẹp */}
    <Box>
      <Typography 
        variant="h6" 
        sx={{ 
          fontSize: '1.1rem',
          fontWeight: 600,
          textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
          letterSpacing: '0.5px'
        }}
      >
        {lesionType}
      </Typography>
    </Box>
  </Box>

  {/* Phần bên phải với các nút điều khiển */}
  <Box sx={{ 
    display: 'flex', 
    alignItems: 'center', 
    gap: 1,
    zIndex: 1 
  }}>
 
    {/* Nút lịch sử với hiệu ứng đẹp */}
    {/* <Tooltip title="Lịch sử hội thoại" arrow>
      <IconButton 
        color="inherit" 
        size="small" 
        onClick={handleHistoryClick}
        disabled={loadingHistory}
        sx={{
          width: 40,
          height: 40,
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          transition: 'all 0.3s ease',
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.2)',
            transform: 'translateY(-2px)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
          },
          '&:active': {
            transform: 'translateY(0)',
          }
        }}
      >
        {loadingHistory ? 
          <CircularProgress color="inherit" size={18} /> : 
          <HistoryIcon sx={{ fontSize: 18 }} />
        }
      </IconButton>
    </Tooltip> */}
  </Box>
</Box>
      
      {/* Menu lịch sử hội thoại */}
      <Menu
        anchorEl={historyAnchorEl}
        open={Boolean(historyAnchorEl)}
        onClose={handleHistoryClose}
        PaperProps={{
          style: {
            maxHeight: 300,
            width: '250px',
          },
        }}
      >
        <MenuItem onClick={handleNewConversation}>
          <Typography variant="body2" fontWeight="bold">Tạo cuộc hội thoại mới</Typography>
        </MenuItem>
        <Divider />
        {conversations.length === 0 && !loadingHistory && (
          <MenuItem disabled>
            <Typography variant="body2">Không có lịch sử hội thoại</Typography>
          </MenuItem>
        )}
        
        {loadingHistory && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
        
        {conversations.map((conv) => (
          <MenuItem 
            key={conv.id} 
            onClick={() => handleSelectConversation(conv.id)}
            selected={conv.id === conversationId}
            sx={{
              '&.Mui-selected': {
                backgroundColor: 'rgba(63, 81, 181, 0.08)',
                borderLeft: '3px solid',
                borderLeftColor: 'primary.main',
                '&:hover': {
                  backgroundColor: 'rgba(63, 81, 181, 0.12)',
                }
              },
              '&:hover': {
                backgroundColor: 'rgba(0, 0, 0, 0.04)',
              },
              '&.Mui-selected .MuiTypography-root': {
                fontWeight: 'bold',
                color: 'primary.main'
              }
            }}
          >
            <Box sx={{ width: '100%' }}>
              <Typography variant="body2" fontWeight={conv.id === conversationId ? 'bold' : 'normal'}>
                {conv.lesion_type}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {format(new Date(conv.created_at), 'dd/MM/yyyy HH:mm', { locale: viLocale })}
              </Typography>
            </Box>
          </MenuItem>
        ))}
      </Menu>
  
      {/* Khu vực tin nhắn */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          p: 2,
          bgcolor: 'background.default',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
        }}
      >
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            content={message.content}
            isUser={message.isUser}
          />
        ))}
        {/* {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )} */}
        <div ref={messagesEndRef} />
      </Box>
  
      {/* Form nhập tin nhắn */}
      <Box
        component="form"
        onSubmit={handleSubmit}
        sx={{ 
          display: 'flex', 
          flexDirection: 'column',
          gap: 1, 
          p: 1,
          borderTop: '1px solid',
          borderColor: 'divider',
          bgcolor: 'background.paper',
          flexShrink: 0,
          position: 'relative',
        }}
      >
       
        {/* Khung input với nút địa cầu và nút gửi */}
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <TextField
            fullWidth
            variant="outlined"
            size="small"
            placeholder="Nhập câu hỏi của bạn..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={loading}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: '20px',
              }
            }}
          />
          
          {/* Nút Internet nhỏ gọn ở giữa */}
          <Tooltip title={enhanceKnowledge ? "Tắt tìm kiếm Internet" : "Bật tìm kiếm Internet"}>
            <IconButton
              onClick={() => setEnhanceKnowledge(!enhanceKnowledge)}
              sx={{
                width: 40,
                height: 40,
                borderRadius: '50%',
                backgroundColor: enhanceKnowledge 
                  ? 'rgba(76, 175, 80, 0.15)'
                  : 'rgba(0, 0, 0, 0.08)',
                border: enhanceKnowledge 
                  ? '2px solid rgba(76, 175, 80, 0.4)'
                  : '2px solid rgba(0, 0, 0, 0.12)',
                transition: 'all 0.3s ease-in-out',
                '&:hover': {
                  transform: 'scale(1.1)',
                  backgroundColor: enhanceKnowledge 
                    ? 'rgba(76, 175, 80, 0.25)'
                    : 'rgba(0, 0, 0, 0.12)',
                }
              }}
            >
              <PublicIcon 
                fontSize="small" 
                sx={{ 
                  color: enhanceKnowledge ? '#4caf50' : '#666',
                  transition: 'color 0.3s ease-in-out'
                }} 
              />
              
            </IconButton>
          </Tooltip>
          
          {/* Nút gửi được làm đẹp */}
          <Button
            type="submit"
            variant="contained"
            disabled={loading || !inputValue.trim()}
            sx={{ 
              minWidth: '80px',
              height: '40px',
              borderRadius: '20px',
              px: 2,
              py: 1,
              background: 'linear-gradient(45deg, #3f51b5 30%, #303f9f 90%)',
              boxShadow: '0 3px 5px 2px rgba(63, 81, 181, .3)',
              '&:hover': {
                background: 'linear-gradient(45deg, #303f9f 30%, #1a237e 90%)',
                transform: 'translateY(-2px)',
                boxShadow: '0 6px 10px 4px rgba(63, 81, 181, .3)',
              },
              '&:active': {
                transform: 'translateY(0)',
              },
              '&:disabled': {
                background: '#cccccc',
                color: '#666',
                transform: 'none',
                boxShadow: 'none',
              },
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
            startIcon={<SendIcon />}
          >
            {loading ? <CircularProgress size={16} color="inherit" /> : 'Send'}
          </Button>
        </Box>
      </Box>
      
      <CrawlVisualizationPopup
        open={showCrawlVisualization}
        onClose={() => setShowCrawlVisualization(false)}
      />
    </Box>
  );
};

export default ChatInterface;