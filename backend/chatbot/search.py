import requests
from bs4 import BeautifulSoup
from googlesearch import search
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def google_search(query, num_results=1):
    """
    Search Google for the given query and return top result URLs
    """
    try:
        logger.info(f"Searching Google for: {query}")
        # Thay đổi tham số num_results sang num_results
        search_results = search(query, num=num_results)
        urls = []
        for url in search_results:
            urls.append({"url": url})
            if len(urls) >= num_results:
                break
        
        logger.info(f"Found {len(urls)} results")
        return urls
    except Exception as e:
        logger.error(f"Google search error: {str(e)}")
        return []