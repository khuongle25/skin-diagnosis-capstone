import React, { useState, useEffect } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import DOMPurify from 'dompurify';
import './ChatMessage.css'; // Đảm bảo import file CSS

const ChatMessage = ({ content, isUser }) => {
  const [formattedHtml, setFormattedHtml] = useState('');

  // Xử lý markdown thành HTML
  useEffect(() => {
    if (!isUser && content) {
      // Xử lý từng phần định dạng
      let html = content;
      
      // Xóa dòng này vì nó có thể gây lỗi với định dạng
      // html = html.replace(/\*\*(?=\S)(.+?)(?<=\S)\*\*/g, '** $1 **');
      
      // Format bold (chữ đậm)
      html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      
      // Format italic (chữ nghiêng)
      html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
      
      // Format bullet points và đảm bảo xuống dòng
      html = html.replace(/^\* (.*?)$/gm, '<li>$1</li>');
      html = html.replace(/^- (.*?)$/gm, '<li>$1</li>');
      
      // Bọc các thẻ li liên tiếp trong một thẻ ul
      html = html.replace(/(<li>.*?<\/li>)(\s*)(?=<li>)/g, '$1');
      html = html.replace(/(?<!<\/ul>)(<li>.*?<\/li>)(?!<li>)/g, '<ul>$1</ul>');
      html = html.replace(/(?<!<\/ul>)(<li>.*<\/li>)(?!<li>)/g, '<ul>$1</ul>');
      
      // Đảm bảo xuống dòng phù hợp
      html = html.replace(/\n\n/g, '<br/><br/>');
      html = html.replace(/\n/g, '<br/>');
      
      // Đảm bảo ul và li có khoảng cách
      html = html.replace(/<ul>/g, '<ul style="margin-left: 20px; margin-bottom: 10px;">');
      html = html.replace(/<li>/g, '<li style="margin-bottom: 8px;">');
      
      // Xử lý các bullet point đặc biệt có chữ đậm
      html = html.replace(/^\* \*\*(.*?)\*\*:/gm, '<li><strong>$1:</strong></li>');
      
      // Sanitize HTML
      const sanitizedHtml = DOMPurify.sanitize(html);
      
      setFormattedHtml(sanitizedHtml);
    }
  }, [content, isUser]);

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Paper
        elevation={1}
        sx={{
          p: 2,
          maxWidth: '75%',
          bgcolor: isUser ? 'primary.light' : 'grey.100',
          color: isUser ? 'white' : 'text.primary',
          borderRadius: 2,
        }}
      >
        {isUser ? (
          <Typography variant="body1">{content}</Typography>
        ) : (
          <div className="message-content bot-message" dangerouslySetInnerHTML={{ __html: formattedHtml }} />
        )}
      </Paper>
    </Box>
  );
};

export default ChatMessage;