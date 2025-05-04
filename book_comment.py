from sanic import Sanic
from sanic.response import json as res_json, file
import time
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 线程冲突
from concurrent.futures import ThreadPoolExecutor
import asyncio
import parse

app = Sanic("mySanic")
executor = None

@app.before_server_start
async def init_executor(app, loop):
    app.ctx.executor = ThreadPoolExecutor(max_workers=4)

@app.after_server_stop
async def shutdown_executor(app, loop):
    app.ctx.executor.shutdown(wait=True)

# 确保目录存在
os.makedirs("upload", exist_ok=True)
os.makedirs("sentiment-analysis", exist_ok=True)


# class CommentWordCloud:
#     """词云类"""
#     _cached_stopwords = None  # 类级缓存
#
#     def __init__(self, label):
#         self.stopword_file_names = ['cn_all_stopwords.txt', 'baidu_stopwords.txt', 'scu_stopwords.txt', 'hit_stopwords.txt',
#                                'cn_stopwords.txt', 'my_stopwords.txt']  # 停用词文件名
#         self.width = 1600
#         self.height = 1200
#         self.max_words = 300
#         self.background_color = 'white'
#         self.scale = 2
#         self.collocations = False
#         self.label = label
#
#     def load_comments(self):
#         """加载评论"""
#         try:
#             all_comment = ""
#             file = self.label + ".jsonl"
#             path = os.path.join("upload", file)
#             with open(path, 'r', encoding="utf-8") as comment_file:
#                 for line_num, line in enumerate(comment_file, 1):
#                     line = line.strip()
#                     if not line:
#                         continue
#                     try:
#                         data = json.loads(line)
#                         comment = data.get("content", "")
#                         all_comment += comment + " "
#                     except json.JSONDecodeError:
#                         print(f"JSON解析失败（第{line_num}行）")
#                     except Exception as e:
#                         print(f"处理第{line_num}行时出错: {e}")
#
#             if not all_comment.strip():
#                 print("无有效评论内容")
#                 return None
#
#             return all_comment
#         except FileNotFoundError:
#             print(f"文件 {path} 未找到")
#             return None
#         except Exception as e:
#             print(f"发生未预期错误: {e}")
#             return None
#
#     def load_stopwords(self):
#         """加载停用词"""
#         if CommentWordCloud._cached_stopwords is None:
#             stopwords = set()
#             for file in self.stopword_file_names:
#                 full_path = os.path.join('stopwords', file)
#                 try:
#                     with open(full_path, 'r', encoding='utf-8') as f:
#                         stopwords.update(line.strip() for line in f)
#                 except FileNotFoundError:
#                     print(f"未找到停用词文件 '{full_path}'，已跳过。")
#             return stopwords
#         return self._cached_stopwords.copy()
#
#
#
#     def make_wordcloud(self):
#         """生成词云"""
#
#         # 中文分词及过滤
#         all_comments = self.load_comments()
#         words = jieba.lcut(all_comments)
#         stopwords = self.load_stopwords()
#         filtered_words = [word for word in words if len(word) > 1 and word not in stopwords and word.strip()]
#         cut_text = " ".join(filtered_words)
#
#         # 跨平台字体设置
#         system = platform.system()
#         if system == 'Windows':
#             font_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'simsun.ttc')
#         elif system == 'Darwin':
#             font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
#         else:
#             font_path = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'
#
#         # 生成词云
#         wordcloud = WordCloud(
#             width=self.width,
#             height=self.height,
#             font_path=font_path,
#             max_words=self.max_words,
#             background_color='white',
#             scale=2,
#             collocations=False
#         ).generate(cut_text)
#
#         plt.figure(figsize=(10, 8))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis("off")
#
#         # 保存词云
#         output_dir = os.path.join(os.getcwd(), "sentiment-analysis")
#         output_path = os.path.join(output_dir, str(self.label)+".png")
#         wordcloud.to_file(output_path)
#         print(f"词云图片已保存至: {output_path}")


# 评论分析
def analyser(id):
    try:
        in_path = os.path.join("upload")
        out_path = os.path.join("sentiment-analysis")
        comment_analyser = parse.Comment_analyser(in_path, out_path, id)
        comment_analyser.make_analyse()
    except Exception as e:
        print(f"影评分析失败：{str(e)}")


# 上传电影数据文件接口
@app.route("/v1/movie/crawled/upload", methods=['POST'])
async def upload(request):
    try:
        file = request.files.get('file')
        if not file:
            return res_json({"code": 0, "message": "文件未上传"}, ensure_ascii=False)

        # 验证文件类型
        filename = file.name
        _, ext = os.path.splitext(filename)
        if ext.lower() != '.json':
            return res_json({"code": 0, "message": "文件格式错误"}, ensure_ascii=False)

        # 生成唯一文件名
        now_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
        id = now_time + "_" + os.path.splitext(filename)[0]
        filename = id + ".jsonl"
        path = os.path.join("upload", filename)

        # 保存文件
        with open(path, 'wb') as f:
            f.write(file.body)

        # 提交后台任务
        app.ctx.executor.submit(analyser, id)  # 直接提交不等待

        # 生成下载链接（label 即图片文件名前缀）
        download_link = f"/v1/image/download?filename={id}.zip"

        return res_json({
            "code": 1,
            "msg": "上传成功，分析任务已提交",
            "data": {
                "name": filename,
                "download_url": download_link  # 新增下载链接
            }
        }, ensure_ascii=False)

    except Exception as e:
        return res_json({"code": 0, "msg": f"服务器错误: {str(e)}"}, ensure_ascii=False)


@app.route("/v1/image/download", methods=['GET'])
async def download_image(request):
    """下载生成压缩包图片"""
    try:
        filename = request.args.get("filename")
        if not filename:
            return res_json({"code": 0, "msg": "缺少 filename 参数"}, ensure_ascii=False)

        # 安全路径验证
        safe_path = os.path.abspath(os.path.join("sentiment-analysis", filename))
        if not safe_path.startswith(os.path.abspath("sentiment-analysis")):
            return res_json({"code": 0, "msg": "非法文件路径"}, ensure_ascii=False)

        # 检查文件是否存在
        if not os.path.exists(safe_path):
            return res_json({"code": 0, "msg": "文件不存在"}, ensure_ascii=False)

        # 返回文件并触发下载
        return await file(
            safe_path,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        return res_json({"code": 0, "msg": f"服务器错误: {str(e)}"}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, workers=4, debug=False)