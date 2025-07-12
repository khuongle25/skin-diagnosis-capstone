import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, Paper, Typography, CircularProgress, Fade, Zoom,
  List, ListItem, ListItemIcon, ListItemText, IconButton,
  Collapse, Grow, Slide, Chip, Tooltip, LinearProgress
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import LinkIcon from '@mui/icons-material/Link';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import DownloadIcon from '@mui/icons-material/Download';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import InfoIcon from '@mui/icons-material/Info';
import CrawlEventService from '../services/sseService';
import './CrawlVisualizationPopup.css';

const validSources = ['enhancement_vi', 'enhancement_en', 'initialization'];

const CrawlVisualizationPopup = ({ open, onClose, mode = 'enhance', onComplete }) => {
  const [eventQueue, setEventQueue] = useState([]);
  const [events, setEvents] = useState([]);
  const [totalUrls, setTotalUrls] = useState({ enhancement_vi: 0, enhancement_en: 0 });
  const [scrapedUrls, setScrapedUrls] = useState({ enhancement_vi: 0, enhancement_en: 0 });
  const [completedSources, setCompletedSources] = useState(new Set());
  const [visualizeDone, setVisualizeDone] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [scrapedEvents, setScrapedEvents] = useState({
    enhancement_vi: {},
    enhancement_en: {}
  });
  const eventsEndRef = useRef(null);
  const crawlServiceRef = useRef(null);
  const processingRef = useRef(false);
  const eventsRef = useRef([]);

  useEffect(() => { eventsRef.current = events; }, [events]);

  // Reset state khi mở popup
  useEffect(() => {
    if (!open) return;
    setEventQueue([]);
    setEvents([]);
    setTotalUrls(0);
    setScrapedUrls(0);
    setCompletedSources(new Set());
    setVisualizeDone(false);

    crawlServiceRef.current = new CrawlEventService(
      (event) => {
        console.log('RECEIVED EVENT', event.event_type, event.source, event.id);
        if (event.source && !validSources.includes(event.source)) return;
        setEventQueue(queue => {
          // Chống duplicate bằng event.id
          const isDuplicate =
            queue.some(e => e.id === event.id) ||
            eventsRef.current.some(e => e.id === event.id);
          return isDuplicate ? queue : [...queue, event];
        });
      },
      () => setIsConnected(true),
      () => setIsConnected(false)
    );
    crawlServiceRef.current.connect();

    return () => {
      if (crawlServiceRef.current) crawlServiceRef.current.disconnect();
    };
  }, [open, mode]);

  // Reset state khi đóng popup
  useEffect(() => {
    if (!open) {
      setEventQueue([]);
      setEvents([]);
      setTotalUrls(0);
      setScrapedUrls(0);
      setCompletedSources(new Set());
      setVisualizeDone(false);
    }
  }, [open]);

  // Xử lý tuần tự từng event trong hàng đợi
  useEffect(() => {
    if (!open) return;
    if (processingRef.current) return;
    if (eventQueue.length === 0) return;

    processingRef.current = true;

    const processNext = async () => {
      if (eventQueue.length === 0) {
        processingRef.current = false;
        return;
      }
      setEventQueue(queue => {
        const sorted = [...queue].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const event = sorted[0];
        if (!event) return sorted.slice(1);
      
        let added = false;
        setEvents(prev => {
          if (prev.some(e => e.id === event.id)) return prev;
          added = true;
          return [...prev, event];
        });
      
        // Chỉ tăng scrapedUrls nếu event được thêm vào events
        if (event.event_type === 'urls_found' && added) {
          setTotalUrls(prev => ({
            ...prev,
            [event.source]: event.data.count || 0
          }));
        }
        if (event.event_type === 'url_scraped' && added) {
          setScrapedUrls(prev => ({
            ...prev,
            [event.source]: (prev[event.source] || 0) + 1
          }));
          setScrapedEvents(prev => ({
            ...prev,
            [event.source]: {
              ...prev[event.source],
              [event.data.index]: event
            }
          }));
        }
        if (event.event_type === 'scraping_complete' && added) {
          setCompletedSources(prev => {
            const next = new Set(prev);
            next.add(event.source);
            return next;
          });
        }
        return sorted.slice(1);
      });
      setTimeout(processNext, 10);
    };

    processNext();
  }, [eventQueue, open]);

  useEffect(() => {
    validSources.forEach(source => {
      if (
        totalUrls[source] > 0 &&
        Object.keys(scrapedEvents[source]).length === totalUrls[source]
      ) {
        const sortedEvents = Object.values(scrapedEvents[source])
          .sort((a, b) => a.data.index - b.data.index);
        // Có thể setEvents([...events, ...sortedEvents]) hoặc visualize theo sortedEvents
      }
    });
  }, [scrapedEvents, totalUrls]);

  // Tính toán tiến trình
  const total = Object.values(totalUrls).reduce((a, b) => a + b, 0);
  const scraped = Object.values(scrapedUrls).reduce((a, b) => a + b, 0);
  const progress = total > 0 ? Math.min((scraped / total) * 100, 100) : 0;
  
  // Kiểm tra hoàn thành
  useEffect(() => {
    if (
      total > 0 &&
      scraped >= total &&
      completedSources.has('enhancement_vi') &&
      completedSources.has('enhancement_en')
    ) {
      setVisualizeDone(true);
      setTimeout(() => {
        if (onComplete) onComplete();
        if (onClose) onClose();
      }, 1200);
    }
  }, [scrapedUrls, totalUrls, completedSources, onComplete, onClose]);

  // Auto-scroll
  useEffect(() => {
    if (eventsEndRef.current) {
      eventsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events]);

  if (!open) return null;

  return (
    <Slide direction="left" in={open} mountOnEnter unmountOnExit>
      <Paper 
        elevation={4}
        sx={{
          position: 'fixed',
          right: 20,
          top: 80,
          width: mode === 'initialization' ? 500 : 450,
          maxHeight: '80vh',
          zIndex: 9999,
          overflow: 'hidden',
          borderRadius: 2,
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 4px 20px rgba(0,0,0,0.15)'
        }}
      >
        {/* Header */}
        <Box 
          className="gradient-header"
          sx={{ 
            p: 2, 
            color: 'white',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <DownloadIcon sx={{ mr: 1, animation: 'pulse 1.5s infinite ease-in-out' }} />
            <Typography variant="subtitle1">
              Quá trình tìm kiếm dữ liệu
            </Typography>
          </Box>
          <Box>
            <IconButton 
              size="small" 
              color="inherit" 
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
            <IconButton 
              size="small" 
              color="inherit"
              onClick={onClose}
            >
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>

        {/* Connection status */}
        <Grow in={!isConnected}>
          <Box sx={{ 
            p: 1, 
            bgcolor: 'warning.light', 
            display: isConnected ? 'none' : 'flex', 
            alignItems: 'center', 
            justifyContent: 'center' 
          }}>
            <CircularProgress size={14} sx={{ mr: 1 }} />
            <Typography variant="caption">Đang kết nối...</Typography>
          </Box>
        </Grow>

        {/* Progress bar */}
        {totalUrls > 0 && (
          <Box sx={{ px: 2, pt: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption">Tiến độ: {Math.round(progress)}%</Typography>
              <Typography variant="caption">{scrapedUrls}/{totalUrls} nguồn</Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={progress} 
              sx={{ 
                height: 8, 
                borderRadius: 1,
                '& .MuiLinearProgress-bar': {
                  transition: 'transform 0.5s ease-out'
                }
              }} 
            />
          </Box>
        )}

        {/* Content */}
        <Collapse in={expanded} timeout="auto">
          <Box 
            sx={{ 
              maxHeight: '60vh', 
              overflowY: 'auto',
              p: 1,
              pt: 2
            }}
          >
            <List dense>
              {events.length === 0 ? (
                <ListItem>
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    width: '100%', 
                    p: 2 
                  }}>
                    <CircularProgress size={30} />
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Đang chuẩn bị tìm kiếm...
                    </Typography>
                  </Box>
                </ListItem>
              ) : (
                events.map((event, index) => {
                  if (!event) return null;
                  let icon, text, secondaryText, chipColor, chipText;
                  
                  switch (event.event_type) {
                    case 'search_started':
                      icon = <SearchIcon color="primary" />;
                      text = `Tìm kiếm: ${event.data.query.substring(0, 25)}...`;
                      secondaryText = `Yêu cầu ${event.data.num_results} kết quả`;
                      chipColor = 'default';
                      chipText = 'BẮT ĐẦU';
                      break;
                    case 'urls_found':
                      icon = <LinkIcon color="info" />;
                      text = `Tìm thấy ${event.data.count} nguồn dữ liệu`;
                      secondaryText = null;
                      chipColor = 'info';
                      chipText = 'TÌM THẤY';
                      break;
                    case 'scraping_url':
                      icon = <CircularProgress size={20} />;
                      text = `Đang trích xuất nguồn ${event.data.index}/${event.data.total}`;
                      secondaryText = event.data.domain;
                      chipColor = 'warning';
                      chipText = 'ĐANG XỬ LÝ';
                      break;
                    case 'url_scraped':
                      icon = event.data.success 
                        ? <CheckCircleIcon color="success" /> 
                        : <ErrorIcon color="error" />;
                      text = event.data.success 
                        ? `Trích xuất thành công (${Math.round(event.data.text_length/1000)}KB)` 
                        : `Lỗi: ${event.data.reason || 'Không xác định'}`;
                      secondaryText = event.data.domain;
                      chipColor = event.data.success ? 'success' : 'error';
                      chipText = event.data.success ? 'THÀNH CÔNG' : 'LỖI';
                      break;
                    case 'scraping_complete':
                      icon = <CheckCircleIcon color="success" />;
                      text = `Hoàn tất: ${event.data.total_scraped}/${event.data.total_attempted} nguồn`;
                      secondaryText = `Hoàn thành lúc ${new Date().toLocaleTimeString()}`;
                      chipColor = 'success';
                      chipText = 'HOÀN TẤT';
                      break;
                    default:
                      icon = <InfoIcon />;
                      text = `Sự kiện: ${event.event_type}`;
                      secondaryText = null;
                      chipColor = 'default';
                      chipText = 'INFO';
                  }
                  
                  return (
                    <Zoom 
                      in={true} 
                      key={event.id}
                      style={{ 
                        transitionDelay: `${index * 100}ms`,
                        transformOrigin: 'center'
                      }}
                    >
                      <ListItem sx={{ 
                        mb: 0.5, 
                        bgcolor: 'background.paper',
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'divider',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                          transform: 'translateY(-2px)'
                        }
                      }}>
                        <ListItemIcon sx={{ minWidth: 40 }}>
                          {icon}
                        </ListItemIcon>
                        <ListItemText 
                          primary={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="body2" noWrap sx={{ mr: 1 }}>
                                {text}
                              </Typography>
                              <Chip 
                                label={chipText} 
                                color={chipColor} 
                                size="small"
                                sx={{ height: 20, '& .MuiChip-label': { px: 0.5, fontSize: '0.6rem' } }}
                              />
                            </Box>
                          }
                          secondary={secondaryText}
                          primaryTypographyProps={{ variant: 'body2' }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                      </ListItem>
                    </Zoom>
                  );
                })
              )}
              <div ref={eventsEndRef} />
            </List>
          </Box>
        </Collapse>
      </Paper>
    </Slide>
  );
};

export default CrawlVisualizationPopup;