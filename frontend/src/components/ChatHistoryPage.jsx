import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Box, Grid, Paper, 
  List, ListItem, ListItemText, ListItemButton, 
  Divider, CircularProgress, Button
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useNavigate } from 'react-router-dom';
import { getUserConversations } from '../services/chatService';
import { useAuth } from '../contexts/AuthContext';
import ChatInterface from '../components/ChatInterface';
import { format } from 'date-fns';
import NavBar from '../components/NavBar';
import getLesionDescription from '../services/LesionDescriptions';

const ChatHistoryPage = () => {
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const { user } = useAuth();
  const navigate = useNavigate();
  const [selectedId, setSelectedId] = useState(null);

  const getDisplayLesionName = (lesionType) => {
    if (!lesionType) return 'Loại không xác định';
        const lesionInfo = getLesionDescription(lesionType);
        return lesionInfo?.name || lesionType;
  };

  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const response = await getUserConversations(user.uid);
        console.log("API Response:", response);
        
        if (Array.isArray(response)) {
          setConversations(response);
        } else if (response && Array.isArray(response.conversations)) {
          setConversations(response.conversations);
        } else {
          console.error("Định dạng dữ liệu không đúng:", response);
          setConversations([]);
        }
      } catch (err) {
        console.error('Lỗi khi tải lịch sử hội thoại:', err);
        setError('Không thể tải lịch sử hội thoại. Vui lòng thử lại sau.');
      } finally {
        setLoading(false);
      }
    };

    fetchConversations();
  }, [user]);
  

  const handleSelectConversation = (conversation) => {
    const convId = conversation.id || conversation.conversation_id;
    console.log("Selected ID:", convId);
    setSelectedId(convId);
    setSelectedConversation(conversation);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Không rõ ngày';
    try {
      return format(new Date(dateString), 'dd/MM/yyyy HH:mm');
    } catch (e) {
      return dateString;
    }
  };

  return (
    <>
      <NavBar />
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Button 
            startIcon={<ArrowBackIcon />} 
            onClick={() => navigate('/home')}
            sx={{ mb: 3 }}
            variant="outlined"
          >
            Quay lại trang chủ
          </Button>

          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Typography color="error" sx={{ my: 2 }}>
              {error}
            </Typography>
          ) : (
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Paper 
                  elevation={3} 
                  sx={{ height: '700px', overflow: 'auto', borderRadius: 2 }}
                >
                  <List sx={{ py: 0 }}>
                    {conversations.length === 0 ? (
                      <ListItem>
                        <ListItemText primary="Bạn chưa có cuộc hội thoại nào" />
                      </ListItem>
                    ) : (
                      conversations.map((conversation) => (
                        <React.Fragment key={conversation.id || conversation.conversation_id}>
                          <ListItemButton 
                            onClick={() => handleSelectConversation(conversation)}
                            selected={selectedId === (conversation.id || conversation.conversation_id)}
                            sx={{ 
                              '&.Mui-selected': {
                                backgroundColor: 'rgba(29, 60, 232, 0.08)', // Màu nhạt hơn
                                borderLeft: '4px solid',
                                borderLeftColor: 'primary.main',
                                '&:hover': {
                                  backgroundColor: 'rgba(63, 81, 181, 0.12)',
                                }
                              },
                              '&:hover': {
                                backgroundColor: 'rgba(0, 0, 0, 0.04)',
                              },
                              '&.Mui-selected .MuiListItemText-primary': {
                                fontWeight: 'bold',
                                color: 'primary.main'
                              }
                            }}
                          >
                            <ListItemText 
                              primary={getDisplayLesionName(conversation.lesion_type)}
                              secondary={`Cập nhật: ${formatDate(conversation.updated_at)}`}
                            />
                          </ListItemButton>
                          <Divider />
                        </React.Fragment>
                      ))
                    )}
                  </List>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={8}>
                <Paper 
                  elevation={3} 
                  sx={{ 
                    height: '700px', 
                    display: 'flex', 
                    flexDirection: 'column',
                    borderRadius: 2,
                    overflow: 'hidden'
                  }}
                >
                  {selectedConversation ? (
                    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <ChatInterface
                        lesionType={getDisplayLesionName(selectedConversation.lesion_type)} // Thay đổi ở đây
                        conversationId={selectedConversation.id || selectedConversation.conversation_id}
                      />
                    </Box>
                  ) : (
                    <Box 
                      sx={{ 
                        display: 'flex', 
                        flexDirection: 'column',
                        justifyContent: 'center', 
                        alignItems: 'center',
                        height: '100%',
                        p: 3
                      }}
                    >
                      <Typography variant="h6" color="text.secondary" align="center">
                        Chọn một cuộc hội thoại từ danh sách bên trái để xem
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Grid>
            </Grid>
          )}
        </Box>
      </Container>
    </>
  );
};

export default ChatHistoryPage;