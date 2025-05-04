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
import logging
import traceback

# 初始化模块日志器
logger = logging.getLogger(__name__)


class CommentWordCloud:
    """词云生成类"""

    def __init__(self):
        self.width = 1600
        self.height = 1200
        self.max_words = 300
        self.background_color = 'white'
        self.scale = 2
        self.collocations = False
        logger.debug("词云生成器初始化完成")

    def make_wordcloud(self, id, out_path, movie_name, comments, comment_count, stopwords):
        """生成带标题的词云"""
        try:
            if not comments:
                logger.warning(f"{id} 无法生成词云：无有效评论内容")
                return None

            logger.info(f"{id} 开始处理词云（{comment_count}条评论）...")

            # 中文分词处理
            words = jieba.lcut(comments)
            filtered_words = [word for word in words if len(word) > 1 and word not in stopwords and word.strip()]
            cut_text = " ".join(filtered_words)
            logger.debug(f"{id} 分词后有效词汇量：{len(filtered_words)}")

            # 字体配置（跨平台）
            system = platform.system()
            if system == 'Windows':
                font_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'simsun.ttc')
            elif system == 'Darwin':
                font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
            else:
                font_path = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'
            logger.debug(f"{id} 使用字体：{font_path}")

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

            # 添加标题
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
            plt.rcParams["axes.unicode_minus"] = False
            title_text = f'《{movie_name}》评论词云分析\n（共{comment_count}条有效评论）'
            plt.title(title_text, fontsize=18, pad=25, fontweight='bold', color='#2B2B2B', loc='center')

            # 保存图片
            path = os.path.join(out_path, id + "_wordcloud.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            logger.info(f"{id} 词云生成成功：{path}")

            return path
        except Exception as e:
            logger.error(f"{id} 词云生成失败：{str(e)}\n{traceback.format_exc()}")
            return None


class CommentHistogram:
    """情感分布图生成类"""

    def make_piechart(self, id, out_path, movie_name, comment_count, positives, negatives):
        """生成情感分布饼状图"""
        try:
            if comment_count == 0:
                logger.warning(f"{id} 无法生成饼图：无有效评论数据")
                return None

            logger.info(f"{id} 开始生成情感分布图...")

            # 配置可视化参数
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
            plt.rcParams["axes.unicode_minus"] = False
            plt.figure(figsize=(10, 8), dpi=120)

            # 构造数据
            categories = ['正面评论', '负面评论']
            values = [positives, negatives]
            colors = ['#7BC8F6', '#FF7F7F']
            explode = (0.05, 0)

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

            # 美化标签
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)

            # 添加标题和图例
            plt.title(f'《{movie_name}》评论情感分布\n总计 {comment_count} 条有效评论',
                      fontsize=16, pad=20, fontweight='bold')
            plt.legend(wedges, categories, title="情感分类", loc="best")

            # 保存图片
            path = os.path.join(out_path, id + "_histogram.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            logger.info(f"{id} 情感分布图生成成功：{path}")

            return path
        except Exception as e:
            logger.error(f"{id} 情感分布图生成失败：{str(e)}\n{traceback.format_exc()}")
            return None


class CommentSemanticAnalyser:
    """情感分析引擎"""

    _cached_analysis = None

    def __init__(self):
        if CommentSemanticAnalyser._cached_analysis is None:
            logger.info("正在初始化情感分析模型...")
            try:
                CommentSemanticAnalyser._cached_analysis = pipeline(
                    Tasks.text_classification,
                    'iic/nlp_structbert_sentiment-classification_chinese-base'
                )
                logger.info("情感分析模型加载完成")
            except Exception as e:
                logger.error(f"模型初始化失败：{str(e)}\n{traceback.format_exc()}")
                raise
        self.analysis = CommentSemanticAnalyser._cached_analysis
        self.positive = 0
        self.negative = 0

    def analyze_sentiment(self, comment_count, comment_list, id):
        """带进度跟踪的情感分析"""
        if comment_count == 0:
            logger.warning(f"{id} 没有需要处理的评论数据")
            return 0, 0

        logger.info(f"{id} 开始情感分析（{comment_count}条评论）...")
        try:
            with tqdm(total=comment_count, desc=f"分析 {id}", unit="comment") as pbar:
                for idx, comment in enumerate(comment_list):
                    try:
                        result = self.analysis(input=comment)
                        sorted_scores = sorted(
                            zip(result['labels'], result['scores']),
                            key=lambda x: x[0] == '正面',
                            reverse=True
                        )

                        if sorted_scores[0][1] >= sorted_scores[1][1]:
                            self.positive += 1
                        else:
                            self.negative += 1

                        # 每100条记录一次进度
                        if (idx + 1) % 100 == 0:
                            logger.debug(f"{id} 已处理{idx + 1}条，正面率{self.positive / (idx + 1):.2%}")

                        pbar.update(1)
                    except Exception as e:
                        logger.warning(f"{id} 第{idx + 1}条评论分析失败：{str(e)}")
                        continue

            logger.info(f"{id} 情感分析完成，正面：{self.positive}，负面：{self.negative}")
            return self.positive, self.negative
        except Exception as e:
            logger.error(f"{id} 情感分析异常终止：{str(e)}\n{traceback.format_exc()}")
            return 0, 0


class Comment_analyser:
    """评论分析主类"""

    _cached_stopwords = None

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
        self.stopword_file_names = ['cn_all_stopwords.txt', 'baidu_stopwords.txt',
                                    'scu_stopwords.txt', 'hit_stopwords.txt',
                                    'cn_stopwords.txt', 'my_stopwords.txt']
        self.comment_wordcloud = CommentWordCloud()
        self.comment_histogram = CommentHistogram()
        self.comment_semantic_analyser = CommentSemanticAnalyser()
        self.histogram_path = None
        self.wordcloud_path = None
        logger.info(f"初始化分析器 {id}")
        self.load_data()
        self.stopwords = self.load_stopwords()

    def load_data(self):
        """加载并解析数据"""
        try:
            path = os.path.join(self.in_path, self.id + ".json")
            logger.info(f"{self.id} 正在加载数据文件：{path}")

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

            logger.info(f"{self.id} 数据加载完成，共{self.comment_count}条有效评论")
        except Exception as e:
            logger.error(f"{self.id} 数据加载失败：{str(e)}\n{traceback.format_exc()}")
            raise

    def load_stopwords(self):
        """加载停用词表"""
        if Comment_analyser._cached_stopwords is None:
            logger.info("正在加载停用词表...")
            stopwords = set()
            for file in self.stopword_file_names:
                full_path = os.path.join('./stopwords', file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        stopwords.update(line.strip() for line in f)
                    logger.debug(f"已加载停用词表：{file}")
                except FileNotFoundError:
                    logger.warning(f"未找到停用词文件：{full_path}")
                except Exception as e:
                    logger.error(f"加载停用词文件{file}失败：{str(e)}")
            Comment_analyser._cached_stopwords = stopwords
            logger.info(f"停用词表加载完成，总计{len(stopwords)}个停用词")
        return Comment_analyser._cached_stopwords

    def create_analysis_archive(self):
        """打包分析结果"""
        logger.info(f"{self.id} 开始打包分析结果...")

        required_files = [
            self.histogram_path,
            self.wordcloud_path
        ]

        missing_files = [f for f in required_files if not f]
        if missing_files:
            logger.error(f"{self.id} 打包失败，缺少文件：{missing_files}")
            return None

        zip_filename = f"{self.id}.zip"
        zip_path = os.path.join(self.out_path, zip_filename)

        try:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for path in required_files:
                    zipf.write(path, os.path.basename(path))
                    logger.debug(f"{self.id} 已添加文件到压缩包：{path}")

            logger.info(f"{self.id} 分析结果已打包至：{zip_path}")
            return zip_path
        except PermissionError:
            logger.error(f"{self.id} 权限被拒绝：无法写入 {zip_path}")
        except Exception as e:
            logger.error(f"{self.id} 打包失败：{str(e)}\n{traceback.format_exc()}")
        return None

    def make_analyse(self):
        """执行完整分析流程"""
        logger.info(f"{self.id} 开始分析流程...")
        try:
            # 情感分析
            self.positives, self.negatives = self.comment_semantic_analyser.analyze_sentiment(
                self.comment_count, self.comment_list, self.id
            )

            # 生成图表
            self.histogram_path = self.comment_histogram.make_piechart(
                self.id, self.out_path, self.movie_name,
                self.comment_count, self.positives, self.negatives
            )

            self.wordcloud_path = self.comment_wordcloud.make_wordcloud(
                self.id, self.out_path, self.movie_name,
                self.comments, self.comment_count, self.stopwords
            )

            # 打包结果
            zip_path = self.create_analysis_archive()

            if zip_path:
                logger.info(f"{self.id} 分析流程成功完成")
                return zip_path
            return None
        except Exception as e:
            logger.error(f"{self.id} 分析流程异常终止：{str(e)}\n{traceback.format_exc()}")
            raise


if __name__ == '__main__':
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("analysis.log"),
            logging.StreamHandler()
        ]
    )

    try:
        os.makedirs("../data", exist_ok=True)
        os.makedirs("../output", exist_ok=True)
        in_path = os.path.join("../data")
        out_path = os.path.join("../output")
        logger.info("启动测试分析任务...")

        comment_analyser = Comment_analyser(in_path, out_path, "movie")
        result = comment_analyser.make_analyse()

        if result:
            logger.info(f"测试任务成功，结果文件：{result}")
        else:
            logger.warning("测试任务未生成有效结果")
    except Exception as e:
        logger.error(f"测试任务失败：{str(e)}\n{traceback.format_exc()}")