import requests
from bs4 import BeautifulSoup
import re
import json
import time

# 统一请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

movie_info = {}


for i in range(0, 250, 25):
    response = requests.get(f"https://movie.douban.com/top250?start={i}", headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        movie_list = soup.find("ol", {"class": "grid_view"})

        if movie_list:
            all_li = movie_list.find_all("li")

            for li in all_li:
                # 获取电影名称
                movie_name_element = li.find("span", {"class": "title"})
                if not movie_name_element:
                    continue

                movie_name = movie_name_element.get_text(strip=True)

                # 获取电影ID
                try:
                    link_element = li.find("div", {"class": "pic"}).find("a")
                    movie_url = link_element["href"]
                    movie_id = re.search(r"subject/(\d+)/", movie_url).group(1)
                except:
                    continue

                print(f"正在爬取: {movie_name} ({movie_id})")

                # 存储当前电影的评论
                current_comments = []


                for j in range(0, 300, 20):
                    review_url = f"https://movie.douban.com/subject/{movie_id}/comments?start={j}&limit=20&status=P&sort=new_score"
                    review_response = requests.get(review_url, headers=headers)

                    if review_response.status_code == 200:
                        review_soup = BeautifulSoup(review_response.text, "html.parser")

                        # 查找所有评论项
                        review_items = review_soup.find_all("div", {"class":"comment-item"})

                        for review_item in review_items:
                            # 提取评论内容
                            comment_element = review_item.find("span", {"class": "short"})

                            if comment_element:
                                # 获取评论文本并清理
                                comment_text = comment_element.get_text(strip=True)
                                current_comments.append(comment_text)

                        # 避免请求过快
                        time.sleep(0.5)

                # 存储当前电影的所有评论
                if current_comments:
                    movie_info[movie_name] = current_comments
    else:
        print(f"请求失败，请求码{response.status_code}")

# 打印并保存结果
print(f"共爬取 {len(movie_info)} 部电影的评论")
with open("movie_info.json", "w", encoding="utf-8") as f:
    json.dump(movie_info, f, ensure_ascii=False, indent=4)