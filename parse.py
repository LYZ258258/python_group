import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud
import platform
import zipfile
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from tqdm import tqdm


class CommentWordCloud:
    """词云类"""

    def __init__(self):
        # 可视化参数
        self.width = 1600
        self.height = 1200
        self.max_words = 300
        self.background_color = 'white'
        self.scale = 2
        self.collocations = False

    def make_wordcloud(self, id, out_path, movie_name, comments, comment_count, stopwords):
        """生成带标题的词云"""
        if not comments:
            print("无法生成词云：无有效评论内容")
            return

        # 中文分词处理
        words = jieba.lcut(comments)
        filtered_words = [word for word in words if len(word) > 1 and word not in stopwords and word.strip()]
        cut_text = " ".join(filtered_words)

        # 字体配置（跨平台）
        system = platform.system()
        if system == 'Windows':
            font_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'simsun.ttc')
        elif system == 'Darwin':
            font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
        else:
            font_path = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'

        # 生成词云对象
        wordcloud = WordCloud(
            width=self.width,
            height=self.height,
            font_path=font_path,
            max_words=self.max_words,
            background_color=self.background_color,
            scale=self.scale,
            collocations=self.collocations
        ).generate(cut_text)

        # 可视化设置
        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        # 添加增强型标题
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]  # Ubuntu 中文字体名称
        plt.rcParams["axes.unicode_minus"] = False
        title_text = f'《{movie_name}》评论词云分析\n（共{comment_count}条有效评论）'
        plt.title(title_text,
                  fontsize=18,
                  pad=25,
                  fontweight='bold',
                  color='#2B2B2B',
                  loc='center')

        # 保存图片
        path = os.path.join(out_path, id + "_wordcloud.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

        return path


class CommentHistogram:
    """饼状图类"""

    def make_piechart(self, id, out_path, movie_name, comment_count, positives, negatives):
        """生成情感分布饼状图"""
        # 数据有效性校验
        if comment_count == 0:
            print("警告：没有可用的评论数据")
            return

        # 配置可视化参数
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]  # Ubuntu 中文字体名称
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(10, 8), dpi=120)

        # 构造数据
        categories = ['正面评论', '负面评论']
        values = [positives, negatives]
        colors = ['#7BC8F6', '#FF7F7F']
        explode = (0.05, 0)  # 突出显示第一部分

        # 创建饼图
        wedges, texts, autotexts = plt.pie(
            values,
            labels=categories,
            colors=colors,
            autopct='%.1f%%',
            startangle=90,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 12, 'color': 'black'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )

        # 美化百分比标签
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        # 添加标题
        plt.title(f'《{movie_name}》评论情感分布\n总计 {comment_count} 条有效评论',
                 fontsize=16, pad=20, fontweight='bold')

        # 添加图例
        plt.legend(wedges,
                  categories,
                  title="情感分类",
                  loc="best",
                  bbox_to_anchor=(1, 0.5, 0.5, 0.5))

        # 保存图片
        path = os.path.join(out_path, id + "_histogram.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

        return path


class CommentSemanticAnalyser:
    """情感分析类"""
    _cached_analysis = None

    def __init__(self):
        if CommentSemanticAnalyser._cached_analysis is None:
            CommentSemanticAnalyser._cached_analysis = pipeline(
                Tasks.text_classification,
                'iic/nlp_structbert_sentiment-classification_chinese-base'
            )
        self.analysis = CommentSemanticAnalyser._cached_analysis
        self.positive = 0
        self.negative = 0

    def analyze_sentiment(self, comment_count, comment_list, id):
        """显示带强制刷新的进度条"""
        if comment_count == 0:
            print("警告：没有需要处理的评论数据！")
            return

        try:
            with tqdm(total=comment_count, desc=f"分析 {id} ing", unit="comment") as pbar:
                for comment in comment_list:
                    try:
                        # 执行情感分析
                        result = self.analysis(input=comment)

                        # 解析模型输出
                        sorted_scores = sorted(
                            zip(result['labels'], result['scores']),
                            key=lambda x: x[0] == '正面',
                            reverse=True
                        )

                        # 构建结果记录
                        if sorted_scores[0][1] >= sorted_scores[1][1]:
                            self.positive += 1
                        else:
                            self.negative += 1

                        pbar.update(1)

                    except Exception as e:
                        pbar.write(f"分析失败: {str(e)}")
                        continue

            print(f"\n情感分析完成")
            return self.positive, self.negative

        except Exception as e:
            print(f"程序异常终止: {str(e)}")


class Comment_analyser:
    _cached_stopwords = None  # 类级缓存

    def __init__(self, in_path, out_path, id):
        self.in_path = in_path
        self.out_path = out_path
        self.id = id
        self.movie_name = ""
        self.comment_count = 0
        self.comment_list = []
        self.comments = ""
        self.positives = 0
        self.negatives = 0
        self.stopword_file_names = ['cn_all_stopwords.txt', 'baidu_stopwords.txt', 'scu_stopwords.txt',
                                    'hit_stopwords.txt', 'cn_stopwords.txt', 'my_stopwords.txt']
        self.comment_wordcloud = CommentWordCloud()
        self.comment_histogram = CommentHistogram()
        self.comment_semantic_analyser = CommentSemanticAnalyser()
        self.histogram_path = None
        self.wordcloud_path = None
        self.load_data()
        self.stopwords = self.load_stopwords()

    def load_data(self):
        """加载并解析数据"""
        try:
            path = os.path.join(self.in_path, self.id + ".json")
            print(f"正在加载数据文件: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                movie_info = json.load(f)
                self.movie_name = movie_info.get("movie_title", "未知电影")
                raw_list = movie_info.get("comment_list", [])

                for comment in raw_list:
                    content = comment.get("comment_content", "").strip()
                    if content:
                        self.comment_count += 1
                        self.comment_list.append(content)
                        self.comments += content

        except Exception as e:
            raise RuntimeError(f"数据加载失败: {str(e)}")

    def load_stopwords(self):
        """加载停用词"""
        if Comment_analyser._cached_stopwords is None:
            stopwords = set()
            for file in self.stopword_file_names:
                full_path = os.path.join('stopwords', file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        stopwords.update(line.strip() for line in f)
                except FileNotFoundError:
                    print(f"未找到停用词文件 '{full_path}'，已跳过。")
            Comment_analyser._cached_stopwords = stopwords
        return Comment_analyser._cached_stopwords

    def create_analysis_archive(self):

        # 验证图片存在性
        required_files = [
            self.histogram_path,
            self.wordcloud_path
        ]

        missing_files = [f for f in required_files if not f]
        if missing_files:
            print("打包失败：缺少以下文件")
            print("\n".join(missing_files))
            return None

        # 生成文件名
        zip_filename = f"{self.id}.zip"
        zip_path = os.path.join(self.out_path, zip_filename)

        try:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for path in required_files:
                    zipf.write(path, os.path.basename(path))

            print(f"分析结果已打包至：{zip_path}")
            return zip_path
        except PermissionError:
            print(f"权限被拒绝：无法写入 {zip_path}")
        except Exception as e:
            print(f"创建压缩包时发生错误：{str(e)}")
        return None

    def make_analyse(self):

        # 情感分析
        self.positives, self.negatives = self.comment_semantic_analyser.analyze_sentiment(self.comment_count, self.comment_list, self.id)

        # 绘制饼状图
        self.histogram_path =  self.comment_histogram.make_piechart(self.id, self.out_path, self.movie_name, self.comment_count, self.positives, self.negatives)

        # 绘制词云
        self.wordcloud_path = self.comment_wordcloud.make_wordcloud(self.id, self.out_path, self.movie_name, self.comments, self.comment_count, self.stopwords)

        # 压缩图片
        self.create_analysis_archive()


if __name__ == '__main__':

    try:
        # 确保目录存在
        os.makedirs("data", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        in_path = os.path.join("data")
        out_path = os.path.join("output")
        comment_analyser = Comment_analyser(in_path, out_path, "movie")
        comment_analyser.make_analyse()
    except Exception as e:
        print(f"影评分析失败：{str(e)}")
