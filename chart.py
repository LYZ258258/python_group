import os
import matplotlib.pyplot as plt
import json
import jieba
from wordcloud import WordCloud
import platform

movie_path = os.getcwd() + "/data/" + "movie.json"
comment_path = os.getcwd() + "/data/" + "comments.jsonl"
emotion_path = os.getcwd() + "/data/" + "comment_emotion.json"


class CommentWordCloud:
    """词云类"""

    def __init__(self, comment_path):
        self.comment_path = comment_path                # 评论文件名
        self.stopword_file_names = ['cn_all_stopwords.txt', 'baidu_stopwords.txt', 'scu_stopwords.txt', 'hit_stopwords.txt',
                               'cn_stopwords.txt', 'my_stopwords.txt']  # 停用词文件名
        self.width = 1600
        self.height = 1200
        self.max_words = 300
        self.background_color = 'white'
        self.scale = 2
        self.collocations = False

    def load_comments(self):
        """加载评论"""
        # file_path = os.path.join(os.getcwd(), "data", self.file_name)
        try:
            all_comment = ""
            with open(self.comment_path, 'r', encoding="utf-8") as comment_file:
                for line_num, line in enumerate(comment_file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        comment = data.get("content", "")
                        all_comment += comment + " "
                    except json.JSONDecodeError:
                        print(f"JSON解析失败（第{line_num}行）")
                    except Exception as e:
                        print(f"处理第{line_num}行时出错: {e}")

            if not all_comment.strip():
                print("无有效评论内容")
                return None

            return all_comment
        except FileNotFoundError:
            print(f"文件 {self.comment_path} 未找到")
            return None
        except Exception as e:
            print(f"发生未预期错误: {e}")
            return None

    def load_stopwords(self):
        """加载停用词"""
        stopwords = set()
        for file_path in self.stopword_file_names:
            full_path = os.path.join('stopwords', file_path)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    stopwords.update(line.strip() for line in f)
            except FileNotFoundError:
                print(f"未找到停用词文件 '{full_path}'，已跳过。")
        return stopwords

    def make_wordcloud(self):
        """生成词云"""

        # 中文分词及过滤
        all_comments = self.load_comments()
        words = jieba.lcut(all_comments)
        stopwords = self.load_stopwords()
        filtered_words = [word for word in words if len(word) > 1 and word not in stopwords and word.strip()]
        cut_text = " ".join(filtered_words)

        # 跨平台字体设置
        system = platform.system()
        if system == 'Windows':
            font_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'simsun.ttc')
        elif system == 'Darwin':
            font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
        else:
            font_path = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'

        # 生成词云
        wordcloud = WordCloud(
            width=self.width,
            height=self.height,
            font_path=font_path,
            max_words=self.max_words,
            background_color='white',
            scale=2,
            collocations=False
        ).generate(cut_text)

        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        # 保存词云
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "wordcloud.png")
        wordcloud.to_file(output_path)  # 修正此处变量名错误
        print(f"词云图片已保存至: {output_path}")

        plt.show()


class CommentHistogram:
    """柱状图类"""

    def __init__(self):
        # 数据处理
        self.movie_name = None
        self.comments = 0
        self.positives = 0
        self.negatives = 0
        self.data_collect()

    def data_collect(self):
        """数据处理"""
        try:
            # 读取电影名称
            with open(movie_path, 'r', encoding='utf-8') as movie_file:
                movie_data = json.load(movie_file)
                self.movie_name = movie_data.get("title", "未知电影")

            # 统计评论总数
            with open(comment_path, 'r', encoding='utf-8') as comment_file:
                self.comments = sum(1 for _ in comment_file)

            # 统计情感分布
            with open(emotion_path, 'r', encoding='utf-8') as emotion_file:
                emotions = json.load(emotion_file)
                for emotion in emotions:
                    try:
                        is_positive = emotion.get("is_positive")
                        if is_positive == 1:
                            self.positives += 1
                        elif is_positive == 0:  # 明确处理0值
                            self.negatives += 1
                    except json.JSONDecodeError:
                        print(f"JSON解析失败：{emotion}")
                    except Exception as e:
                        print(f"处理情感数据时出错：{e}")
        except FileNotFoundError as e:
            print(f"文件未找到：{e}")
        except Exception as e:
            print(f"数据收集时发生错误：{e}")

    def make_histogram(self):
        """生成情感分布柱状图"""
        # 数据有效性校验
        if self.comments == 0:
            print("警告：没有可用的评论数据")
            return
        if (self.positives + self.negatives) == 0:
            print("警告：没有有效的情感分析数据")
            return

        # 配置可视化参数
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 更清晰的中文字体
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 12
        plt.figure(figsize=(14, 8), dpi=120)  # 提高分辨率

        # 构造数据
        categories = ['正面评论', '负面评论']
        total = self.positives + self.negatives
        percentages = [self.positives / total * 100, self.negatives / total * 100] if total > 0 else [0, 0]
        values = [self.positives, self.negatives]

        # 创建渐变色条
        colors = ['#7BC8F6', '#FF7F7F']

        # 绘制立体柱状图
        bars = plt.bar(categories, values,
                       color=colors,
                       edgecolor='black',
                       linewidth=1.5,
                       width=0.7,
                       alpha=0.9)

        # 添加数据标签
        for i, (v, p) in enumerate(zip(values, percentages)):
            label_text = f"{v}\n({p:.1f}%)" if total > 0 else "0\n(0%)"
            plt.text(i, v + max(values) * 0.02,
                     label_text,
                     ha='center',
                     va='bottom',
                     fontsize=12,
                     fontweight='bold',
                     color=colors[i])

        # 图表装饰
        plt.title(f'《{self.movie_name}》评论情感分析\n总计 {self.comments} 条有效评论',
                  fontsize=18, pad=20, fontweight='bold')
        plt.ylabel('评论数量', fontsize=14, labelpad=10)
        plt.ylim(0, max(values) * 1.3)

        # 添加网格线
        plt.grid(axis='y',
                 alpha=0.4,
                 linestyle='--',
                 color='gray')

        # 优化坐标轴样式
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')

        # 保存高清图片
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "histogram.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图表已保存至：{save_path}")

        # 显示图表
        # plt.show()



if __name__ == '__main__':
    # 生成高质量词云
    cloud = CommentWordCloud(comment_path)
    cloud.make_wordcloud()

    # 生成精美柱状图
    histogram = CommentHistogram()
    histogram.make_histogram()
