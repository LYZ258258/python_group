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

# 获取模块日志器（继承主程序配置）
logger = logging.getLogger(__name__)


class CommentWordCloud:
    """词云生成器"""

    def __init__(self):
        self.config = {
            'width': 1600,
            'height': 1200,
            'max_words': 300,
            'background_color': 'white',
            'scale': 2,
            'collocations': False
        }
        logger.debug("词云生成器初始化")

    def generate(self, task_id, output_dir, title, text, count, stopwords):
        """生成词云图片"""
        try:
            logger.info(f"[{task_id}] 开始生成词云")

            # 分词处理
            words = jieba.lcut(text)
            filtered = [w for w in words if len(w) > 1 and w not in stopwords]
            logger.debug(f"[{task_id}] 有效词汇量: {len(filtered)}")

            # 字体配置
            system = platform.system()
            font_path = {
                'Windows': os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'simsun.ttc'),
                'Darwin': '/System/Library/Fonts/STHeiti Medium.ttc',
                'Linux': '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'
            }.get(system)
            logger.debug(f"[{task_id}] 使用字体: {font_path}")

            # 生成词云
            wc = WordCloud(font_path=font_path, **self.config).generate(" ".join(filtered))

            # 可视化设置
            plt.figure(figsize=(12, 10))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(
                f'《{title}》评论词云分析\n（共{count}条有效评论）',
                fontsize=18,
                pad=25,
                fontweight='bold',
                color='#2B2B2B'
            )

            # 保存文件
            output_path = os.path.join(output_dir, f"{task_id}_wordcloud.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"[{task_id}] 词云已保存: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"[{task_id}] 词云生成失败: {str(e)}", exc_info=True)
            return None


class SentimentVisualizer:
    """情感分析可视化"""

    def generate_pie(self, task_id, output_dir, title, total, positive, negative):
        """生成饼图"""
        try:
            logger.info(f"[{task_id}] 开始生成情感分布图")

            plt.figure(figsize=(10, 8), dpi=120)
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
            plt.rcParams["axes.unicode_minus"] = False

            # 数据准备
            labels = ['正面', '负面']
            sizes = [positive, negative]
            colors = ['#7BC8F6', '#FF7F7F']
            explode = (0.05, 0)

            # 绘制饼图
            wedges, texts, autotexts = plt.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                shadow=True,
                textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'black', 'linewidth': 1}
            )

            # 样式优化
            for autotext in autotexts:
                autotext.set_fontweight('bold')

            plt.title(
                f'《{title}》情感分布\n总计 {total} 条评论',
                fontsize=16,
                pad=20,
                fontweight='bold'
            )
            plt.legend(wedges, labels, title="情感分类", loc="best")

            # 保存文件
            output_path = os.path.join(output_dir, f"{task_id}_sentiment.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"[{task_id}] 情感分布图已保存: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"[{task_id}] 饼图生成失败: {str(e)}", exc_info=True)
            return None


class SentimentAnalyzer:
    """情感分析引擎"""

    _model = None

    def __init__(self):
        if not SentimentAnalyzer._model:
            logger.info("初始化情感分析模型...")
            try:
                SentimentAnalyzer._model = pipeline(
                    Tasks.text_classification,
                    'iic/nlp_structbert_sentiment-classification_chinese-base'
                )
                logger.info("模型加载成功")
            except Exception as e:
                logger.error("模型初始化失败", exc_info=True)
                raise

    def analyze(self, comments, task_id):
        """执行情感分析"""
        logger.info(f"[{task_id}] 开始分析 {len(comments)} 条评论")
        positive = 0
        negative = 0

        try:
            with tqdm(total=len(comments), desc=f"分析 {task_id}") as pbar:
                for idx, text in enumerate(comments):
                    try:
                        result = self._model(text)
                        label = max(zip(result['labels'], result['scores']), key=lambda x: x[1])[0]
                        positive += 1 if label == '正面' else 0
                        negative += 1 if label == '负面' else 0

                        if (idx + 1) % 100 == 0:
                            logger.debug(f"[{task_id}] 已处理 {idx + 1} 条，正面率 {positive / (idx + 1):.2%}")
                        pbar.update(1)
                    except Exception as e:
                        logger.warning(f"[{task_id}] 第 {idx + 1} 条分析失败: {str(e)}")
            logger.info(f"[{task_id}] 分析完成 | 正面: {positive} | 负面: {negative}")
            return positive, negative
        except Exception as e:
            logger.error(f"[{task_id}] 分析中断: {str(e)}", exc_info=True)
            return 0, 0


class AnalysisPipeline:
    """分析流水线"""

    def __init__(self, input_dir, output_dir, task_id):
        self.task_id = task_id
        self.logger = logging.getLogger(f"SA-Processor.Task.{task_id}")
        self.input_path = os.path.join(input_dir, f"{task_id}.json")
        self.output_dir = output_dir
        self.data = {
            "title": "",
            "comments": [],
            "raw_text": ""
        }
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        """加载停用词表"""
        logger.info(f"[{self.task_id}] 加载停用词表")
        stopwords = set()
        for filename in ['cn_all_stopwords.txt', 'baidu_stopwords.txt']:
            try:
                with open(os.path.join('stopwords', filename), 'r', encoding='utf-8') as f:
                    stopwords.update(line.strip() for line in f)
            except Exception as e:
                logger.warning(f"[{self.task_id}] 停用词文件 {filename} 加载失败: {str(e)}")
        return stopwords

    def load_data(self):
        """加载数据文件"""
        try:
            logger.info(f"[{self.task_id}] 加载数据文件: {self.input_path}")
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data['title'] = data.get('movie_title', '未知电影')
                self.data['comments'] = [
                    c['comment_content'].strip()
                    for c in data.get('comment_list', [])
                    if c['comment_content'].strip()
                ]
                self.data['raw_text'] = " ".join(self.data['comments'])
            logger.info(f"[{self.task_id}] 载入 {len(self.data['comments'])} 条有效评论")
        except Exception as e:
            logger.error(f"[{self.task_id}] 数据加载失败", exc_info=True)
            raise

    def execute(self):
        """执行完整分析流程"""
        try:
            # 情感分析
            analyzer = SentimentAnalyzer()
            positive, negative = analyzer.analyze(self.data['comments'], self.task_id)

            # 可视化
            visualizer = SentimentVisualizer()
            cloud_gen = CommentWordCloud()

            pie_path = visualizer.generate_pie(
                self.task_id,
                self.output_dir,
                self.data['title'],
                len(self.data['comments']),
                positive,
                negative
            )

            wordcloud_path = cloud_gen.generate(
                self.task_id,
                self.output_dir,
                self.data['title'],
                self.data['raw_text'],
                len(self.data['comments']),
                self.stopwords
            )

            # 打包结果
            return self._package_results(pie_path, wordcloud_path)
        except Exception as e:
            logger.error(f"[{self.task_id}] 分析流程异常终止", exc_info=True)
            raise

    def _package_results(self, *paths):
        """打包分析结果"""
        try:
            zip_path = os.path.join(self.output_dir, f"{self.task_id}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for path in paths:
                    if path and os.path.exists(path):
                        zipf.write(path, os.path.basename(path))
            logger.info(f"[{self.task_id}] 结果已打包: {zip_path}")
            return zip_path
        except Exception as e:
            logger.error(f"[{self.task_id}] 打包失败", exc_info=True)
            return None


if __name__ == '__main__':
    # 独立运行时的日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("analysis.log"),
            logging.StreamHandler()
        ]
    )

    try:
        pipeline = AnalysisPipeline("data", "output", "test_task")
        pipeline.load_data()
        result_path = pipeline.execute()
        logging.info(f"测试任务完成: {result_path}")
    except Exception as e:
        logging.error(f"测试失败: {str(e)}", exc_info=True)