import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';

import { 
  Container, Box, Typography, Grid, Paper, Avatar, Switch, FormControlLabel,
  Button, Card, CardContent, CardActions, Alert,
  IconButton, Divider, CircularProgress, TextField,
  Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import PersonIcon from '@mui/icons-material/Person';
import DeleteIcon from '@mui/icons-material/Delete';
import SegmentedImage from './SegmentedImage';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import axios from 'axios';
import { format } from 'date-fns';
import viLocale from 'date-fns/locale/vi';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const UserProfile = () => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [commentDialogOpen, setCommentDialogOpen] = useState(false);
  const [currentPostId, setCurrentPostId] = useState(null);
  const [commentText, setCommentText] = useState('');
  const [comments, setComments] = useState([]);
  const [loadingComments, setLoadingComments] = useState(false);
  const [actionSuccess, setActionSuccess] = useState(null);
  const { user } = useAuth();
  const location = useLocation();

  useEffect(() => {
  const fetchData = () => {
    if (user) {
      fetchUserPosts();
    }
  };

  // Gọi khi component mount hoặc user/location thay đổi
  fetchData();

  // Thêm event listener cho window focus
  const handleFocus = () => fetchData();
  window.addEventListener('focus', handleFocus);
  
  return () => {
    window.removeEventListener('focus', handleFocus);
  };
}, [user, location.pathname]);
  
  // Xóa thông báo thành công sau 3 giây
  useEffect(() => {
    if (actionSuccess) {
      const timer = setTimeout(() => {
        setActionSuccess(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [actionSuccess]);

  useEffect(() => {
    // Thêm sự kiện refresh khi tab được focus lại
    const handleFocus = () => {
      if (user) {
        fetchUserPosts();
      }
    };
   
    // Thêm event listener
    window.addEventListener('focus', handleFocus);
    
    // Cleanup
    return () => {
      window.removeEventListener('focus', handleFocus);
    };
  }, [user]);

  useEffect(() => {
    if (user) {
      fetchUserPosts();
    }
  }, [user, location.pathname]);

  const fetchUserPosts = async () => {
    if (!user) return;
    
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/posts/user/${user.uid}`);
      
      // Log chi tiết hơn để hiểu đúng kiểu dữ liệu
      console.log("===== USER POSTS RAW DATA =====");
      console.log(response.data);
      
      if (response.data && response.data.length > 0) {
        const firstPost = response.data[0];
        console.log("First post details:");
        console.log("is_starred:", firstPost.is_starred); 
        console.log("is_starred type:", typeof firstPost.is_starred);
        console.log("is_starred === true:", firstPost.is_starred === true);
        console.log("is_starred == true:", firstPost.is_starred == true);
        console.log("is_starred === 'True':", firstPost.is_starred === 'True');
        console.log("is_starred == 'True':", firstPost.is_starred == 'True');
      }
      
      setPosts(response.data);
    } catch (error) {
      console.error('Error fetching user posts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleVisibility = async (postId, currentVisibility) => {
    try {
      const response = await axios.put(`${API_URL}/api/posts/${postId}/visibility`, {
        visible: !currentVisibility
      });
      
      // Cập nhật danh sách posts
      setPosts(prevPosts => prevPosts.map(post => {
        if (post.id === postId) {
          return {
            ...post,
            visible: !currentVisibility
          };
        }
        return post;
      }));
      
      setActionSuccess(`Đã ${!currentVisibility ? 'công khai' : 'ẩn'} bài đăng thành công`);
    } catch (error) {
      console.error('Error toggling visibility:', error);
      setError('Không thể cập nhật trạng thái hiển thị. Vui lòng thử lại sau.');
    }
  };

  const handleDeletePost = async (postId) => {
    // Hiển thị xác nhận trước khi xóa
    if (!window.confirm('Bạn có chắc chắn muốn xóa bài đăng này?')) {
      return;
    }
    
    try {
      // Thêm user_id vào query parameter
      await axios.delete(`${API_URL}/api/posts/${postId}`, {
        params: { user_id: user.uid }
      });
      
      // Xóa post khỏi danh sách
      setPosts(prevPosts => prevPosts.filter(post => post.id !== postId));
      setActionSuccess('Xóa bài đăng thành công');
    } catch (error) {
      console.error('Error deleting post:', error);
      setError('Không thể xóa bài đăng. Vui lòng thử lại sau.');
    }
  };

  const handleStarPost = async (postId, isStarred) => {
    // Chuẩn hóa giá trị isStarred để đảm bảo so sánh chính xác
    const isCurrentlyStarred = isStarred === true || isStarred === 'True';
    
    if (!user) return;
  
    // Lưu trạng thái ban đầu để khôi phục nếu có lỗi
    const originalPosts = [...posts];
  
    // Optimistic update UI
    setPosts(prevPosts => prevPosts.map(post => {
      if (post.id === postId) {
        return {
          ...post,
          is_starred: !isCurrentlyStarred,
          star_count: isCurrentlyStarred ? Math.max(0, post.star_count - 1) : (post.star_count + 1)
        };
      }
      return post;
    }));
  
    try {
      if (isCurrentlyStarred) {
        // Remove star
        await axios.delete(`${API_URL}/api/stars/${postId}`, {
          params: { user_id: user.uid }
        });
      } else {
        // Add star
        await axios.post(`${API_URL}/api/stars/`, {
          post_id: postId,
          user_id: user.uid
        });
      }
      
      // Đọc trạng thái từ server để đảm bảo đồng bộ
      const response = await axios.get(`${API_URL}/api/posts/${postId}/starred`, {
        params: { user_id: user.uid }
      });
      
      // Nếu trạng thái từ server khác với UI, update lại
      const serverStarred = response.data.is_starred;
      if ((serverStarred === true || serverStarred === 'True') !== !isCurrentlyStarred) {
        fetchUserPosts();
      }
    } catch (error) {
      console.error('Error updating star status:', error);
      // Khôi phục trạng thái gốc nếu có lỗi
      setPosts(originalPosts);
    }
  };

  const handleOpenCommentDialog = async (postId) => {
    setCurrentPostId(postId);
    setCommentDialogOpen(true);
    setCommentText('');
    
    // Fetch comments
    setLoadingComments(true);
    try {
      const response = await axios.get(`${API_URL}/api/posts/${postId}/comments`);
      setComments(response.data);
    } catch (error) {
      console.error('Error fetching comments:', error);
    } finally {
      setLoadingComments(false);
    }
  };

  const handleCloseCommentDialog = () => {
    setCommentDialogOpen(false);
    setCurrentPostId(null);
  };

  const handleSubmitComment = async () => {
    if (!user || !commentText.trim() || !currentPostId) return;
  
    try {
      const response = await axios.post(`${API_URL}/api/comments/`, {
        post_id: currentPostId,
        user_id: user.uid,
        content: commentText.trim()
      });
  
      // Cập nhật UI với comment mới
      setComments([...comments, response.data]);
      
      // Cập nhật số lượng bình luận trong danh sách posts
      setPosts(prevPosts => prevPosts.map(post => {
        if (post.id === currentPostId) {
          return {
            ...post,
            comment_count: (post.comment_count || 0) + 1
          };
        }
        return post;
      }));
      
      setCommentText('');
    } catch (error) {
      console.error('Error posting comment:', error);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography variant="body1" sx={{ mt: 2 }}>Đang tải bài đăng...</Typography>
      </Container>
    );
  }

  if (!user) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="h6">Vui lòng đăng nhập để xem trang cá nhân</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 8 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center" color="primary" sx={{ mb: 4 }}>
        CÁ NHÂN
      </Typography>
      
      {actionSuccess && (
        <Alert severity="success" sx={{ mb: 2 }}>{actionSuccess}</Alert>
      )}
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}

      {posts.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6">Bạn chưa có bài đăng nào</Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
            Hãy chia sẻ kết quả chẩn đoán để tạo bài đăng mới!
          </Typography>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {posts.map(post => (
            <Grid item xs={12} key={post.id}>
            <Card elevation={3} sx={{
              overflow: 'hidden',
              transition: 'transform 0.2s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
              }
            }}>
              {/* Phần header */}
              <CardContent sx={{ pb: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                      <PersonIcon />
                    </Avatar>
                    <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                      {user.display_name || user.displayName || 'Bạn'} {/* Luôn hiển thị tên người dùng hiện tại vì đây là trang cá nhân */}
                    </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {format(new Date(post.created_at), 'dd/MM/yyyy HH:mm', { locale: viLocale })}
                      </Typography>
                    </Box>
                  </Box>
                  
                  {/* Công khai/Riêng tư switch - chỉ hiển thị trong UserProfile */}
                    <FormControlLabel
                      control={
                        <Switch
                          checked={post.visible}
                          onChange={() => handleToggleVisibility(post.id, post.visible)}
                          color="primary"
                        />
                      }
                      label={
                        <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          {post.visible ? 
                            <><VisibilityIcon fontSize="small" color="primary" /> Công khai</> : 
                            <><VisibilityOffIcon fontSize="small" color="disabled" /> Riêng tư</>
                          }
                        </Typography>
                      }
                    />
                </Box>
                
                {/* Tiêu đề bài đăng với hiệu ứng */}
                <Typography variant="h6" gutterBottom sx={{ 
                  color: 'primary.dark',
                  fontWeight: 600,
                  pb: 1,
                  borderBottom: '1px solid',
                  borderColor: 'divider'
                }}>
                  {post.title}
                </Typography>
                
                {/* Nội dung bài đăng */}
                <Typography variant="body1" paragraph sx={{ py: 1 }}>
                  {post.content}
                </Typography>
              </CardContent>
              
              {/* Phần ảnh và thông tin chẩn đoán */}
              <Grid container spacing={0}>
                {/* Phần ảnh bên trái */}
                <Grid item xs={12} md={6} sx={{ p: 2, bgcolor: '#f9f9f9' }}>
                  <Box sx={{ 
                    position: 'relative', 
                    borderRadius: 2, 
                    overflow: 'hidden',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
                  }}>
                    <SegmentedImage 
                      imageData={post.image_data}
                      maskData={post.mask_data}
                      height={240}
                    />
                  </Box>
                </Grid>
                
                {/* Phần thông tin chẩn đoán bên phải */}
                <Grid item xs={12} md={6} sx={{ p: 2 }}>
                  <Paper elevation={0} sx={{ 
                    p: 2, 
                    height: '100%', 
                    bgcolor: 'background.paper',
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 2
                  }}>
                    <Box sx={{ mb: 2, pb: 1.5, borderBottom: '1px dashed', borderColor: 'divider' }}>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Kết quả chẩn đoán:
                      </Typography>
                      <Typography variant="h6" color="primary.dark" fontWeight="bold" gutterBottom>
                        {post.diagnosis.label}
                      </Typography>
                      
                      <Typography variant="caption" sx={{ 
                        display: 'inline-block',
                        py: 0.5, 
                        px: 1, 
                        bgcolor: 'primary.light',
                        color: 'white',
                        borderRadius: 1,
                        mt: 0.5
                      }}>
                        Độ tin cậy: {(post.diagnosis.confidence * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                    
                    {post.patient_metadata && Object.keys(post.patient_metadata).length > 0 && (
                      <Box sx={{ pt: 0.5 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Thông tin bệnh nhân:
                        </Typography>
                        
                        <Box sx={{ pl: 1, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                          {post.patient_metadata.age && (
                            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Box component="span" sx={{ fontWeight: 'medium', color: 'text.secondary' }}>Tuổi:</Box> {post.patient_metadata.age}
                            </Typography>
                          )}
                          
                          {post.patient_metadata.gender && (
                            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Box component="span" sx={{ fontWeight: 'medium', color: 'text.secondary' }}>Giới tính:</Box>
                              {post.patient_metadata.gender === 'male' ? 'Nam' : 
                                post.patient_metadata.gender === 'female' ? 'Nữ' : 'Không xác định'}
                            </Typography>
                          )}
                          
                          {post.patient_metadata.location && (
                            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Box component="span" sx={{ fontWeight: 'medium', color: 'text.secondary' }}>Vị trí:</Box> {post.patient_metadata.location}
                            </Typography>
                          )}
                        </Box>
                      </Box>
                    )}
                  </Paper>
                </Grid>
              </Grid>
              
              {/* Phần tương tác (star, comment) */}
              <CardActions sx={{ px: 2, pb: 2, display: 'flex', justifyContent: 'space-between', bgcolor: 'background.paper' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <IconButton 
                  onClick={() => handleStarPost(post.id, !!post.is_starred)}
                  color={!!post.is_starred ? 'primary' : 'default'}
                  disabled={!user}
                  size="small"
                >
                  {!!post.is_starred ? <StarIcon /> : <StarBorderIcon />}
                </IconButton>
                <Typography variant="body2" sx={{ mr: 2 }}>
                  {post.star_count !== undefined ? post.star_count : 0}
                </Typography>             

                <IconButton onClick={() => handleOpenCommentDialog(post.id)} size="small">
                  <ChatBubbleOutlineIcon />
                </IconButton>
                <Typography variant="body2">
                  {typeof post.comment_count === 'number' ? post.comment_count : 
                  typeof post.comment_count === 'string' ? parseInt(post.comment_count) || 0 : 0}
                </Typography>
                </Box>
                
                {/* Nút xóa bài đăng - chỉ hiển thị trong UserProfile */}
                  <Button 
                    variant="outlined" 
                    color="error"
                    size="small"
                    onClick={() => handleDeletePost(post.id)}
                    startIcon={<DeleteIcon />}
                  >
                    Xóa bài đăng
                  </Button>
              </CardActions>
            </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Comment Dialog - giống như trong CommunityFeed */}
      <Dialog 
        open={commentDialogOpen} 
        onClose={handleCloseCommentDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Bình luận</DialogTitle>
        <DialogContent dividers>
          {loadingComments ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              {comments.length === 0 ? (
                <Typography variant="body1" align="center" sx={{ py: 3 }}>
                  Chưa có bình luận nào. Hãy là người đầu tiên bình luận!
                </Typography>
              ) : (
                <Box sx={{ mb: 2 }}>
                  {comments.map(comment => (
                    <Box key={comment.id} sx={{ mb: 2, pb: 2, borderBottom: '1px solid #eee' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main', mr: 1 }}>
                          <PersonIcon fontSize="small" />
                        </Avatar>
                        <Typography variant="subtitle2">
                          {comment.user_display_name || 'Người dùng ẩn danh'}
                        </Typography>
                        <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                          {format(new Date(comment.created_at), 'dd/MM/yyyy HH:mm', { locale: viLocale })}
                        </Typography>
                      </Box>
                      <Typography variant="body1" sx={{ pl: 5 }}>
                        {comment.content}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}

              {user ? (
                <Box sx={{ mt: 2, display: 'flex', alignItems: 'flex-start' }}>
                  <Avatar sx={{ mr: 1.5, mt: 1 }}>
                    <PersonIcon />
                  </Avatar>
                  <TextField
                    fullWidth
                    multiline
                    rows={2}
                    variant="outlined"
                    placeholder="Viết bình luận của bạn..."
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                  />
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary" align="center">
                  Vui lòng đăng nhập để bình luận.
                </Typography>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseCommentDialog}>
            Đóng
          </Button>
          {user && (
            <Button 
              onClick={handleSubmitComment} 
              variant="contained" 
              color="primary"
              disabled={!commentText.trim()}
            >
              Gửi bình luận
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default UserProfile;