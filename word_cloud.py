import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import json
import os

def generate_movie_wordcloud(comments_path,useless_path):
    with open(comments_path, "r", encoding="utf-8") as f:
        info_json = json.load(f)
    with open(useless_path, 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines()
        stop_words += ['', ' ', '\n', '\t']  # 补充空白字符过滤
    for movie_name, comments_list in info_json.items():
        os.makedirs("./image_file/"+movie_name, exist_ok=True)

        # 将多条短评合并为一个字符串
        all_comments = " ".join(comments_list)

        # 使用jieba进行分词
        seg_list = jieba.cut(all_comments)
        seg_text = " ".join(seg_list)

        # 生成词云
        wordcloud = WordCloud(
            font_path='msyh.ttc',
            background_color='white',
            width=800, height=600,
            max_words=100,  # 最多显示100个词
            min_font_size=10,  # 最小字体大小
            collocations=False,  # 不统计重复短语
            stopwords=stop_words
        ).generate(seg_text)

        # 绘制词云图
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # 不显示坐标轴
        savepath = f"./image_file/{movie_name}/《{movie_name}》词云图.png"
        plt.savefig(savepath)
        plt.close()
if __name__ == "__main__":
    generate_movie_wordcloud("test.json", "useless_word.txt")

