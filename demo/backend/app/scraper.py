import logging
import ipaddress
import socket
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import newspaper

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
            
        try:
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback:
                return False
        except socket.gaierror:
            pass
            
        return True
    except Exception:
        return False

def scrape_article(url: str) -> dict:
    result = {"title": "", "text": "", "source": ""}
    
    if not is_safe_url(url):
        logger.warning(f"Unsafe URL detected: {url}")
        return result
    
    # Try newspaper3k first
    try:
        article = newspaper.Article(url, headers=HEADERS)
        article.download()
        article.parse()
        if article.text and len(article.text.strip()) > 100:
            result["title"] = article.title
            result["text"] = article.text.strip()
            result["source"] = urlparse(url).netloc
            return result
    except Exception as e:
        logger.warning("newspaper3k failed to parse URL", exc_info=True)
    
    # Fallback to BeautifulSoup
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        
        title = soup.title.string if soup.title else ""
        result["title"] = title.strip() if title else ""
        result["source"] = urlparse(url).netloc
        
        # Selectors
        selectors = [
            '.fck_detail', # VnExpress
            '.detail-content', # Tuoi Tre
            '.content_detail', # Dan Tri
            '.article__body', # Thanh Nien
            '#main-detail-body', # VietnamNet
            '.detail__content' # Zing News
        ]
        
        content = ""
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator='\n', strip=True)
                break
        
        if not content:
            # General fallback: get all paragraphs
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
        result["text"] = content.strip()
        return result
    except Exception as e:
        logger.warning("BeautifulSoup fallback failed", exc_info=True)
        return result
