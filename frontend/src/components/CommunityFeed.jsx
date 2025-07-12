import React, { useState, useEffect } from 'react';
import { 
  Container, Box, Typography, Grid, Paper, Avatar,
  Button, Card, CardContent, CardActions,
  IconButton, Divider, CircularProgress, TextField,
  Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import PersonIcon from '@mui/icons-material/Person';
import SegmentedImage from './SegmentedImage';

import axios from 'axios';
import { format } from 'date-fns';
import viLocale from 'date-fns/locale/vi';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const CommunityFeed = () => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [commentDialogOpen, setCommentDialogOpen] = useState(false);
  const [currentPostId, setCurrentPostId] = useState(null);
  const [commentText, setCommentText] = useState('');
  const [comments, setComments] = useState([]);
  const [loadingComments, setLoadingComments] = useState(false);
  const { user } = useAuth();

  useEffect(() => {
    fetchPosts();
  }, []);

  const fetchPosts = async () => {
    setLoading(true);
    try {
      const userId = user ? user.uid : '';
      const response = await axios.get(`${API_URL}/api/posts/?user_id=${userId}`);
      setPosts(response.data);
    } catch (error) {
      console.error('Error fetching posts:', error);
      setError('Không thể tải bài đăng. Vui lòng thử lại sau.');
    } finally {
      setLoading(false);
    }
  };

  const handleStarPost = async (postId, isStarred) => {
    isStarred = isStarred === true || isStarred === "true";
    if (!user) return;
  
    try {
      if (isStarred) {
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
  
      // Update posts state với spread operator để tránh lỗi type
      setPosts(prevPosts => prevPosts.map(post => {
        if (post.id === postId) {
          return {
            ...post,
            is_starred: !isStarred,
            star_count: isStarred ? (post.star_count || 1) - 1 : (post.star_count || 0) + 1
          };
        }
        return post;
      }));
    } catch (error) {
      console.error('Error updating star:', error);
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
  
      // Server sẽ trả về comment đã có user_display_name
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

  if (error) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="error">{error}</Typography>
        <Button variant="contained" sx={{ mt: 2 }} onClick={fetchPosts}>
          Thử lại
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 8 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center" color="primary" sx={{ mb: 4 }}>
        CỘNG ĐỒNG
      </Typography>

      {posts.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6">Chưa có bài đăng nào</Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
            Hãy là người đầu tiên chia sẻ kết quả chẩn đoán của bạn!
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
                          {post.user_display_name || 'Người dùng ẩn danh'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {format(new Date(post.created_at), 'dd/MM/yyyy HH:mm', { locale: viLocale })}
                        </Typography>
                      </Box>
                    </Box>
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
                  onClick={() => handleStarPost(post.id, post.is_starred)}
                  color={post.is_starred === true || post.is_starred === 'True' ? 'primary' : 'default'}
                  disabled={!user}
                  size="small"
                >
                  {post.is_starred === true || post.is_starred === 'True' ? <StarIcon /> : <StarBorderIcon />}
                </IconButton>
                    <Typography variant="body2" sx={{ mr: 2 }}>
                      {post.star_count || 0}
                    </Typography>
                    
                    <IconButton onClick={() => handleOpenCommentDialog(post.id)} size="small">
                      <ChatBubbleOutlineIcon />
                    </IconButton>
                    <Typography variant="body2">
                      {post.comment_count || 0}
                    </Typography>
                  </Box>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Comment Dialog */}
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

export default CommunityFeed;