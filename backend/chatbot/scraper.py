import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import os
import time
import re
from typing import List, Optional
from googlesearch import search as google_search
from googleapiclient.discovery import build

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Làm sạch văn bản đã scrape"""
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    # Loại bỏ các ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\s.,;:?!()-]', '', text)
    return text.strip()

def scrape_website(url: str) -> Optional[str]:
    """Scrape nội dung từ một website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"Scraping content from: {url}")
        domain = urlparse(url).netloc
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Loại bỏ các phần không cần thiết
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Ưu tiên các phần có nội dung chính
        main_content = None
        content_elements = soup.select('main, article, .content, #content, .article, .post')
        
        if content_elements:
            main_content = content_elements[0]
            text = main_content.get_text(separator='\n', strip=True)
        else:
            # Nếu không tìm thấy phần nội dung chính, lấy toàn bộ nội dung
            text = soup.get_text(separator='\n', strip=True)
        
        # Làm sạch văn bản
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        text = clean_text(text)
        
        # Lưu nội dung đã scrape vào log (tùy chọn)
        log_dir = "scraper_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Tạo tên file an toàn
        safe_domain = domain.replace('.', '_').replace('/', '_')
        log_file = os.path.join(log_dir, f"{safe_domain}_{hash(url)}.txt")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Domain: {domain}\n")
            f.write("=" * 80 + "\n")
            f.write(text)
        
        logger.info(f"Successfully scraped {len(text)} characters from {domain}")
        return text
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return None

def search_websites(query: str, num_results: int = 3) -> List[str]:
    try:
        search_query = f"{query} medical dermatology skin condition"
        
        urls = []
        # Sử dụng stop=num_results để giới hạn TỔNG SỐ kết quả
        for url in google_search(search_query, stop=num_results):
            urls.append(url)
            
        logger.info(f"Found {len(urls)} URLs for query: {search_query}")
        return urls
    except Exception as e:
        logger.error(f"Error searching for {query}: {str(e)}")
        return []

GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

def google_cse_search(query, num_results=3):
    service = build("customsearch", "v1", developerKey=GOOGLE_CSE_API_KEY)
    res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
    urls = []
    if "items" in res:
        for item in res["items"]:
            urls.append(item["link"])
    return urls

def scrape_websites_for_query(query: str, num_results: int = 3, source: str = "unknown") -> list:
    """Tìm kiếm và scrape nội dung từ các website cho một truy vấn"""
    from googlesearch import search as google_search

    # 1. Bắt đầu tìm kiếm
    add_crawl_event("search_started", {"query": query, "num_results": num_results}, source=source)
    try:
        urls = google_cse_search(f"{query} medical dermatology skin condition", num_results)
    except Exception as e:
        add_crawl_event("urls_found", {"urls": [], "count": 0, "error": str(e)}, source=source)
        return []

    # 2. Tìm thấy URLs
    add_crawl_event("urls_found", {"urls": urls, "count": len(urls)}, source=source)

    scraped_texts = []
    for i, url in enumerate(urls):
        domain = urlparse(url).netloc
        # 3. Đang trích xuất từng nguồn
        add_crawl_event("scraping_url", {
            "url": url,
            "domain": domain,
            "index": i + 1,
            "total": len(urls)
        }, source=source)

        # 4. Thực hiện scrape
        text = scrape_website(url)
        time.sleep(1)
        if text and len(text) > 200:
            scraped_texts.append(text)
            add_crawl_event("url_scraped", {
                "url": url,
                "domain": domain,
                "success": True,
                "text_length": len(text),
                "index": i + 1,
                "total": len(urls)
            }, source=source)
        else:
            add_crawl_event("url_scraped", {
                "url": url,
                "domain": domain,
                "success": False,
                "reason": "insufficient_content",
                "index": i + 1,
                "total": len(urls)
            }, source=source)

    # 5. Hoàn thành
    add_crawl_event("scraping_complete", {
        "total_scraped": len(scraped_texts),
        "total_attempted": len(urls)
    }, source=source)
    return scraped_texts

def add_crawl_event(event_type: str, data: dict, source: str = "unknown") -> None:
    """Gửi event đến SSE server thông qua HTTP request"""
    import datetime
    try:
        payload = {
            "event_type": event_type,
            "data": data,
            "source": source,
            "timestamp": datetime.datetime.utcnow().isoformat()  # Đảm bảo có timestamp
        }
        response = requests.post("http://localhost:8001/publish-event", json=payload)
        logger.info(f"Event sent: {event_type} | source={source} | data={data}")
        if not response.ok:
            print(f"Failed to send event to SSE server: {response.text}")
    except Exception as e:
        print(f"Error sending event to SSE server: {str(e)}")

