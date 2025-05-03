import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from tqdm import tqdm  # 导入进度条库

comment_path = os.getcwd() + "/data/" + "comments.jsonl"
emotion_path = os.getcwd() + "/data/" + "comment_emotion.json"


class CommentSemanticAnalysis:
    def __init__(self):
        # 初始化情感分析模型
        self.analysis = pipeline(
            Tasks.text_classification,
            'iic/nlp_structbert_sentiment-classification_chinese-base'
        )
        self.total_comments = self.count_comments()  # 先统计有效评论总数
        self.comments = self.load_comments()         # 创建评论生成器
        self.header = ['is_positive', 'positive_probs', 'negative_probs']
        self.file = self.initialize_file()
        self.first_record = True

    def count_comments(self):
        """精确统计有效评论数量"""
        count = 0
        try:
            with open(comment_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="统计评论"):
                    try:
                        data = json.loads(line)
                        if data.get("content", "").strip():
                            count += 1
                    except json.JSONDecodeError:
                        continue
            return count
        except FileNotFoundError:
            raise Exception("评论文件未找到，请先运行爬虫程序")

    def load_comments(self):
        """安全加载评论数据（生成器版本）"""
        try:
            with open(comment_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if content := data.get("content", "").strip():
                            yield content
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            raise Exception("评论文件未找到，请先运行爬虫程序")

    def initialize_file(self):
        """初始化结果文件"""
        file = open(emotion_path, 'w', encoding='utf-8')
        file.write('[\n')  # 开始JSON数组
        return file

    def save_progress(self, record):
        """实时保存进度"""
        if self.first_record:
            self.first_record = False
        else:
            self.file.write(',\n')
        json.dump(record, self.file, ensure_ascii=False, indent=4)

    def analyze_sentiment(self):
        """执行情感分析并保存结果"""
        try:
            # 创建进度条
            with tqdm(total=self.total_comments, desc="分析情感", unit="comment") as pbar:
                for comment in self.comments:
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
                        record = {
                            'is_positive': 1 if sorted_scores[0][1] >= sorted_scores[1][1] else 0,
                            'positive_probs': sorted_scores[0][1],
                            'negative_probs': sorted_scores[1][1]
                        }

                        # 保存结果并更新进度
                        self.save_progress(record)
                        pbar.update(1)

                    except Exception as e:
                        pbar.write(f"分析失败: {str(e)}")
                        continue

            print(f"\n分析完成！结果已保存到 {emotion_path}")

        except Exception as e:
            print(f"程序异常终止: {str(e)}")
        finally:
            self.file.write('\n]')
            self.file.close()


if __name__ == "__main__":
    analyzer = CommentSemanticAnalysis()
    analyzer.analyze_sentiment()