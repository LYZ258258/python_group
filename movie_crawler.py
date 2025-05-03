# coding=utf-8
import requests
from bs4 import BeautifulSoup
import time
import random
import json
import os
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import jsonlines

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("douban_crawler.log"),
        logging.StreamHandler()
    ]
)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMENT_PATH = os.path.join(BASE_DIR, "data", "comments.jsonl")
MOVIE_PATH = os.path.join(BASE_DIR, "data", "movie.json")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# 自动创建data文件夹保存数据
os.makedirs(os.path.dirname(MOVIE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(COMMENT_PATH), exist_ok=True)

# 加载配置文件
with open(CONFIG_PATH) as f:
    config = json.load(f)

MOVIE_ID = config.get("movie_id", "36415357")
MAX_PAGES = config.get("max_pages", 10)
REQUEST_INTERVAL = tuple(config.get("request_interval", [3, 8]))
MAX_RETRIES = config.get("max_retries", 3)
TIMEOUT = config.get("timeout", 20)
CONCURRENT_WORKERS = config.get("concurrent_workers", 3)
PROXY_API_URL = config["proxy_api_url"]
PROXY_CREDENTIALS = config["proxy_credentials"]
COOKIE_STR = config.get("cookie", "")

# User-Agent列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.3; rv:109.0) Gecko/20100101 Firefox/122.0'
]


class ProxyManager:
    """代理管理类"""

    def __init__(self):
        self.proxies = []
        self.lock = threading.Lock()
        self.proxies_exhausted = threading.Event()
        self._load_proxies()

    def _load_proxies(self):
        """初始加载并验证代理"""
        logging.info("正在初始化代理池...")
        try:
            resp = requests.get(PROXY_API_URL, timeout=15)
            data = resp.json()
            if data["code"] == 1:
                raw_proxies = [f"{item['ip']}:{item['port']}" for item in data["data"]]
                self.proxies = self._validate_proxies(raw_proxies)
                logging.info(f"代理加载完成，有效代理数：{len(self.proxies)}")
            else:
                logging.error("代理加载失败: %s", data.get("info"))
        except Exception as e:
            logging.error("代理加载异常: %s", str(e))

    def _validate_proxies(self, proxies):
        """验证代理有效性"""
        test_url = "https://www.douban.com/"
        valid_proxies = []

        def check_proxy(proxy):
            try:
                formatted = {
                    "http": f"http://{PROXY_CREDENTIALS['user']}:{PROXY_CREDENTIALS['pass']}@{proxy}",
                    "https": f"http://{PROXY_CREDENTIALS['user']}:{PROXY_CREDENTIALS['pass']}@{proxy}"
                }
                resp = requests.get(
                    test_url,
                    proxies=formatted,
                    timeout=10,
                    headers={"User-Agent": random.choice(USER_AGENTS)}
                )
                if resp.status_code == 200 and "异常请求" not in resp.text:
                    return formatted
            except:
                return None

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(check_proxy, proxies)

        return [p for p in results if p]

    def get_proxy(self):
        """获取代理并打印使用信息"""
        with self.lock:
            if not self.proxies:
                self.proxies_exhausted.set()
                return None
            proxy = random.choice(self.proxies)
            ip_port = proxy['http'].split('@')[-1]
            logging.info("使用代理: %s", ip_port)
            return proxy

    def report_proxy_status(self, proxy, success):
        """更新代理状态"""
        with self.lock:
            if not success and proxy in self.proxies:
                ip_port = proxy['http'].split('@')[-1]
                self.proxies.remove(proxy)
                logging.warning("移除失效代理: %s", ip_port)
                if not self.proxies:
                    self.proxies_exhausted.set()


class EnhancedCrawler:
    """增强型爬虫基类"""

    def __init__(self, proxy_manager):
        self.session = requests.Session()
        self.proxy_manager = proxy_manager
        self.last_request_time = 0
        self.request_lock = threading.Lock()
        self._setup_session()

    def _setup_session(self):
        """初始化会话设置"""
        self.session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
            'Cookie': COOKIE_STR,
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': f'https://movie.douban.com/subject/{MOVIE_ID}/'
        })

    def _request_interval(self):
        """请求间隔控制"""
        with self.request_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < REQUEST_INTERVAL[0]:
                sleep_time = random.uniform(*REQUEST_INTERVAL)
                time.sleep(sleep_time)
            self.last_request_time = time.time()

    def fetch(self, url, params=None):
        """执行请求"""
        if self.proxy_manager.proxies_exhausted.is_set():
            logging.warning("代理池已耗尽，停止请求")
            return None

        for attempt in range(MAX_RETRIES):
            self._request_interval()
            proxy = self.proxy_manager.get_proxy()
            if not proxy:
                return None

            try:
                response = self.session.get(
                    url,
                    params=params,
                    proxies=proxy,
                    timeout=TIMEOUT
                )

                if response.status_code in [403, 429]:
                    logging.warning("触发反爬机制 [%d]", response.status_code)
                    self.proxy_manager.report_proxy_status(proxy, False)
                    continue

                response.raise_for_status()
                return response.text

            except requests.exceptions.RequestException as e:
                logging.warning("请求失败: %s", str(e))
                self.proxy_manager.report_proxy_status(proxy, False)
                time.sleep(2 ** attempt)

        logging.error("请求失败: %s (超过最大重试次数)", url)
        return None


class DoubanMovieCrawler(EnhancedCrawler):
    """电影信息爬取"""

    def crawl(self):
        logging.info("开始爬取电影基本信息...")
        html = self.fetch(f'https://movie.douban.com/subject/{MOVIE_ID}')
        movie_info = {}
        if html:
            movie_info = self.parse_movie(html)
        self.save_movie_info(movie_info)
        return movie_info

    def parse_movie(self, html):
        soup = BeautifulSoup(html, 'lxml')
        return {
            "douban_id": MOVIE_ID,
            "title": self._safe_extract(soup, 'h1 span[property="v:itemreviewed"]'),
            "year": self._extract_year(soup),
            "rating": self._extract_rating(soup),
            "directors": self._extract_people(soup, '导演'),
            "actors": self._extract_people(soup, '主演'),
            "genres": self._extract_genres(soup),
            "summary": self._safe_extract(soup, 'div.related-info div.indent span[property="v:summary"]')
        }

    def _safe_extract(self, soup, selector):
        element = soup.select_one(selector)
        return element.text.strip() if element else None

    def _extract_year(self, soup):
        year_text = self._safe_extract(soup, 'h1 .year')
        return re.search(r'\d{4}', year_text).group() if year_text else None

    def _extract_rating(self, soup):
        rating_str = self._safe_extract(soup, 'strong.ll.rating_num')
        try:
            return float(rating_str) if rating_str else None
        except ValueError:
            return None

    def _extract_people(self, soup, role):
        try:
            span = soup.find('span', text=role)
            return [a.text.strip() for a in span.find_next('span').find_all('a')]
        except AttributeError:
            return []

    def _extract_genres(self, soup):
        return [genre.text for genre in soup.select('span[property="v:genre"]')]

    def save_movie_info(self, data):
        os.makedirs(os.path.dirname(MOVIE_PATH), exist_ok=True)
        with open(MOVIE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info("电影信息已保存至: %s", MOVIE_PATH)


class DoubanCommentCrawler(EnhancedCrawler):
    """评论爬取"""

    def __init__(self, proxy_manager):
        super().__init__(proxy_manager)
        self.comment_count = 0
        self.file_lock = threading.Lock()
        self.stop_event = proxy_manager.proxies_exhausted

    def crawl(self):
        logging.info("开始爬取电影评论...")
        self._init_comment_file()

        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = []
            for page in range(MAX_PAGES):
                if self.stop_event.is_set():
                    logging.warning("代理池耗尽，停止提交新任务")
                    break
                futures.append(executor.submit(self.process_page, page))
                logging.info(f"已提交第 {page + 1}/{MAX_PAGES} 页请求")

            for future in concurrent.futures.as_completed(futures):
                if self.stop_event.is_set():
                    future.cancel()
                    continue
                try:
                    count = future.result()
                    logging.info(f"当前总计评论数: {self.comment_count}")
                except concurrent.futures.CancelledError:
                    pass

        logging.info("评论爬取完成，总计获取 %d 条评论", self.comment_count)
        return self.comment_count

    def _init_comment_file(self):
        """初始化评论文件"""
        with jsonlines.open(COMMENT_PATH, mode='w') as f:
            pass

    def process_page(self, page):
        """处理单个评论页"""
        if self.stop_event.is_set():
            return 0

        params = {'start': page * 20, 'limit': 20, 'sort': 'new_score'}
        html = self.fetch(f'https://movie.douban.com/subject/{MOVIE_ID}/comments', params)
        if not html:
            return 0

        comments = self.parse_comments(html)
        return self._save_comments(comments)

    def parse_comments(self, html):
        """解析评论内容"""
        soup = BeautifulSoup(html, 'lxml')
        comments = []
        for item in soup.select('div.comment-item'):
            try:
                comment = {
                    'user': item.select_one('h3 > span.comment-info > a').text.strip(),
                    'rating': self._extract_rating(item),
                    'time': item.select_one('span.comment-time')['title'],
                    'content': item.select_one('span.short').text.strip(),
                    'votes': int(item.select_one('span.votes').text),
                    'cid': item.get('data-cid', '')
                }
                comments.append(comment)
            except Exception as e:
                logging.debug(f"解析评论失败: {str(e)}")
        logging.info(f"本页解析到 {len(comments)} 条有效评论")
        return comments

    def _extract_rating(self, item):
        rating_tag = item.select_one('span.rating')
        if rating_tag and 'class' in rating_tag.attrs:
            for cls in rating_tag['class']:
                match = re.match(r'allstar(\d{2})', cls)
                if match:
                    return int(match.group(1)) // 10
        return None

    def _save_comments(self, comments):
        """线程安全保存评论"""
        if not comments:
            return 0

        with self.file_lock:
            try:
                with jsonlines.open(COMMENT_PATH, mode='a') as writer:
                    writer.write_all(comments)
                self.comment_count += len(comments)
                logging.info(f"新增 {len(comments)} 条评论，当前总计: {self.comment_count}")
            except Exception as e:
                logging.error(f"保存失败: {str(e)}")
                return 0
        return len(comments)


if __name__ == "__main__":
    proxy_manager = ProxyManager()

    # 爬取电影信息
    movie_crawler = DoubanMovieCrawler(proxy_manager)
    movie_info = movie_crawler.crawl()

    # 爬取评论
    comment_crawler = DoubanCommentCrawler(proxy_manager)
    comment_count = comment_crawler.crawl()

    logging.info("任务完成！电影信息获取%s，共爬取%d条评论",
                 "成功" if movie_info else "失败", comment_count)