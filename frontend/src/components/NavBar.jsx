import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import ChatIcon from '@mui/icons-material/Chat';
import PeopleIcon from '@mui/icons-material/People';
import AccountBoxIcon from '@mui/icons-material/AccountBox';

const NavBar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/signin');
  };

  const handleChatHistoryClick = () => {
    navigate('/chat-history');
  };

  const handleCommunityClick = () => {
    navigate('/community');
  };

  const handleMyPostsClick = () => {
    navigate('/my-posts');
  };

  return (
    <AppBar 
      position="static"
      sx={{
        background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)'
      }}
    >
      <Toolbar>
        <Typography 
          variant="h6" 
          component="div" 
          sx={{ 
            flexGrow: 1, 
            cursor: 'pointer',
            fontWeight: 'bold',
            textShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
          }}
          onClick={() => navigate('/home')}
        >
          SKIN LESION DIAGNOSIS
        </Typography>
        {user ? (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="body1" sx={{ 
              mr: 2,
              fontWeight: '500',
              textShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
            }}>
              Xin chào, {user.display_name || user.email}
            </Typography>
            <Button
              color="inherit"
              startIcon={<ChatIcon />}
              onClick={handleChatHistoryClick}
              sx={{ 
                mr: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 1
                }
              }}
            >
              Lịch sử chat
            </Button>
            
            <Button
              color="inherit"
              startIcon={<PeopleIcon />}
              onClick={handleCommunityClick}
              sx={{ 
                mr: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 1
                }
              }}
            >
              Cộng đồng
            </Button>

            <Button
              color="inherit"
              startIcon={<AccountBoxIcon />}
              onClick={handleMyPostsClick}
              sx={{ 
                mr: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 1
                }
              }}
            >
              Cá nhân
            </Button>
        
            <Button 
              color="inherit" 
              variant="outlined"
              onClick={handleLogout}
              sx={{ 
                border: '2px solid rgba(255, 255, 255, 0.8)',
                borderRadius: 2,
                fontWeight: 'bold',
                px: 3,
                '&:hover': {
                  border: '2px solid white',
                  backgroundColor: 'rgba(255, 255, 255, 0.15)',
                  transform: 'translateY(-1px)',
                  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)'
                },
                transition: 'all 0.2s ease'
              }}
            >
              LOG OUT
            </Button>
          </Box>
        ) : (
          <Button 
            color="inherit" 
            onClick={() => navigate('/signin')}
            sx={{
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderRadius: 1
              }
            }}
          >
            Đăng nhập
          </Button>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default NavBar;