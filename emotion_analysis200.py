from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json


def analysis(comments_path):
    # 读取电影评论数据
    with open(comments_path, "r", encoding='utf-8') as f:
        movie_comment_dic = json.load(f)

    # 初始化情感分析模型（只需要一次）
    semantic_cls = pipeline(Tasks.text_classification, "iic/nlp_structbert_sentiment-classification_chinese-tiny")

    movie_mapdata = {}

    # 遍历每部电影
    for movie_name, comments in movie_comment_dic.items():
        positive_count = 0  # 每部电影单独计数

        # 遍历该电影的所有评论
        for comment in comments:
            try:
                # 进行情感分析
                result = semantic_cls(input=comment)

                # 解析结果
                labels = result["labels"]
                scores = result["scores"]

                # 找到"正面"和"负面"标签对应的分数
                positive_score = next((score for label, score in zip(labels, scores) if label == "正面"), 0)
                negative_score = next((score for label, score in zip(labels, scores) if label == "负面"), 0)

                # 判断为正面的条件：正面分数大于负面分数
                if positive_score > negative_score:
                    positive_count += 1
            except Exception as e:
                print(f"分析电影 '{movie_name}' 的评论时出错: {e}")
                continue

        # 计算情感比例（避免除以0）
        total_comments = len(comments)
        if total_comments > 0:
            positive_level = round(positive_count / total_comments, 2)
            negative_level = round(1 - positive_level, 2)
        else:
            positive_level = 0
            negative_level = 0

        # 保存结果
        movie_mapdata[movie_name] = {
            "positive_level": positive_level,
            "negative_level": negative_level,
            "total_comments": total_comments,
            "positive_count": positive_count
        }

    # 保存结果到JSON文件
    with open("emotion.json", "w", encoding='utf-8') as f:
        json.dump(movie_mapdata, f, ensure_ascii=False, indent=2)
