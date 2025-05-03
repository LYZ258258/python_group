import json
import uuid
import os
from tqdm import tqdm
from sanic import Sanic, response
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pathlib import Path

app = Sanic("SentimentAnalysisService")
app.config.TEMP_DIR = "./temp_analysis"
app.config.MAX_FILE_SIZE = 1024 * 1024 * 10  # 10MB
app.config.ALLOWED_EXTENSIONS = {'jsonl'}

# 初始化情感分析模型（在服务启动时加载）
nlp_pipeline = pipeline(
    Tasks.text_classification,
    'iic/nlp_structbert_sentiment-classification_chinese-base'
)


class AnalysisTask:
    def __init__(self):
        self.tasks = {}
        Path(app.config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    def create_task(self, input_path):
        task_id = str(uuid.uuid4())
        task_dir = Path(app.config.TEMP_DIR) / task_id
        task_dir.mkdir()

        self.tasks[task_id] = {
            "status": "processing",
            "input": str(task_dir / "input.jsonl"),
            "output": str(task_dir / "result.json"),
            "progress": 0.0,
            "error": None
        }

        # 移动上传文件到任务目录
        os.rename(input_path, self.tasks[task_id]["input"])
        return task_id


task_manager = AnalysisTask()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config.ALLOWED_EXTENSIONS


@app.post("/analyze")
async def analyze_file(request):
    # 验证文件上传
    upload_file = request.files.get('file')
    if not upload_file:
        return response.json({"error": "No file uploaded"}, status=400)

    if not allowed_file(upload_file.name):
        return response.json({"error": "Invalid file type"}, status=400)

    # 保存临时文件
    temp_path = Path(app.config.TEMP_DIR) / f"temp_{uuid.uuid4().hex}.jsonl"
    with open(temp_path, "wb") as f:
        f.write(upload_file.body)

    # 创建分析任务
    try:
        task_id = task_manager.create_task(temp_path)
        app.add_task(process_analysis(task_id))
        return response.json({"task_id": task_id})
    except Exception as e:
        return response.json({"error": str(e)}, status=500)


async def process_analysis(task_id):
    task = task_manager.tasks[task_id]
    try:
        # 准备分析数据
        comments = []
        with open(task["input"], "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if content := data.get("content", "").strip():
                        comments.append(content)
                except json.JSONDecodeError:
                    continue

        total = len(comments)
        progress_step = 100 / total if total > 0 else 0

        # 开始分析处理
        with open(task["output"], "w", encoding="utf-8") as out_file:
            out_file.write("[\n")
            first_entry = True

            with tqdm(total=total, desc="Analyzing") as pbar:
                for idx, comment in enumerate(comments):
                    try:
                        result = nlp_pipeline(input=comment)
                        sorted_scores = sorted(
                            zip(result['labels'], result['scores']),
                            key=lambda x: x[0] == '正面',
                            reverse=True
                        )

                        record = {
                            "text": comment,
                            "positive": sorted_scores[0][1],
                            "negative": sorted_scores[1][1]
                        }

                        if not first_entry:
                            out_file.write(",\n")
                        json.dump(record, out_file, ensure_ascii=False)
                        first_entry = False

                        # 更新进度
                        task["progress"] = min(100.0, (idx + 1) * progress_step)
                        pbar.update(1)

                    except Exception as e:
                        print(f"Error processing comment: {str(e)}")
                        continue

            out_file.write("\n]")
            task["status"] = "completed"

    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
    finally:
        # 清理临时文件
        try:
            os.remove(task["input"])
        except:
            pass


@app.get("/status/<task_id>")
async def get_status(request, task_id):
    task = task_manager.tasks.get(task_id)
    if not task:
        return response.json({"error": "Invalid task ID"}, status=404)

    return response.json({
        "status": task["status"],
        "progress": task["progress"],
        "error": task["error"]
    })


@app.get("/download/<task_id>")
async def download_result(request, task_id):
    task = task_manager.tasks.get(task_id)
    if not task:
        return response.json({"error": "Invalid task ID"}, status=404)

    if task["status"] != "completed":
        return response.json({"error": "Analysis not completed"}, status=400)

    if not Path(task["output"]).exists():
        return response.json({"error": "Result file missing"}, status=500)

    return await response.file(task["output"], filename="analysis_result.json")


@app.listener("after_server_stop")
async def cleanup(app, loop):
    # 服务停止时清理临时目录
    for task_id in list(task_manager.tasks.keys()):
        task_dir = Path(app.config.TEMP_DIR) / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=2)