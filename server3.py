import json
import uuid
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from sanic import Sanic, response
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

app = Sanic("SentimentAnalysisService")
app.config.update({
    "TEMP_DIR": "./analysis_tasks",
    "MAX_FILE_SIZE": 10 * 1024 * 1024,  # 10MB
    "ALLOWED_EXTENSIONS": {'jsonl'}
})

# 初始化情感分析模型（保持原有分析逻辑）
sentiment_analysis = pipeline(
    Tasks.text_classification,
    'iic/nlp_structbert_sentiment-classification_chinese-base'
)

class TaskManager:
    def __init__(self):
        self.tasks = {}
        Path(app.config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    def create_task(self, input_file):
        """创建新任务并转移文件"""
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
        
        shutil.move(input_file, self.tasks[task_id]["input"])
        return task_id

task_mgr = TaskManager()

def validate_file(filename):
    """文件验证（保持原有格式要求）"""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() == 'jsonl')

@app.post("/upload")
async def upload_file(request):
    """文件上传接口（保持原有输入格式）"""
    # 验证文件
    upload_file = request.files.get('file')
    if not upload_file:
        return response.json({"error": "未上传文件"}, status=400)
    
    if not validate_file(upload_file.name):
        return response.json({"error": "仅支持.jsonl格式文件"}, status=400)

    if len(upload_file.body) > app.config.MAX_FILE_SIZE:
        return response.json({"error": "文件超过10MB限制"}, status=413)

    # 保存临时文件
    temp_path = Path(app.config.TEMP_DIR) / f"upload_{uuid.uuid4().hex}.jsonl"
    with open(temp_path, "wb") as f:
        f.write(upload_file.body)

    # 创建任务
    try:
        task_id = task_mgr.create_task(temp_path)
        app.add_task(process_task(task_id))
        return response.json({
            "task_id": task_id,
            "status": "/task/status/" + task_id,
            "result": "/task/result/" + task_id
        })
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

async def process_task(task_id):
    """处理任务（保持原有输出格式）"""
    task = task_mgr.tasks[task_id]
    
    try:
        # 统计有效行数
        total = 0
        with open(task["input"], "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): total +=1
        
        # 初始化输出文件
        with open(task["output"], "w", encoding="utf-8") as out_file:
            out_file.write("[\n")
            first_item = True
            
            with tqdm(total=total, desc="分析进度") as pbar:
                with open(task["input"], "r", encoding="utf-8") as in_file:
                    for idx, line in enumerate(in_file):
                        try:
                            data = json.loads(line)
                            text = data.get("content", "").strip()
                            if not text:
                                continue
                            
                            # 执行情感分析（保持原有逻辑）
                            result = sentiment_analysis(text)
                            labels = result['labels']
                            scores = result['scores']
                            
                            # 构建结果记录（保持原有输出格式）
                            record = {
                                "is_positive": int(scores[labels.index('正面')] >= 0.5),
                                "positive_probs": scores[labels.index('正面')],
                                "negative_probs": scores[labels.index('负面')]
                            }
                            
                            # 写入结果
                            if not first_item:
                                out_file.write(",\n")
                            json.dump(record, out_file, ensure_ascii=False)
                            first_item = False
                            
                            # 更新进度
                            task["progress"] = (idx + 1) / total * 100
                            pbar.update(1)
                            
                        except Exception as e:
                            continue
            
            out_file.write("\n]")
            task["status"] = "completed"
            
    except Exception as e:
        task.update(status="error", error=str(e))
    finally:
        # 清理输入文件
        try:
            os.remove(task["input"])
        except:
            pass

@app.get("/task/status/<task_id>")
async def get_status(request, task_id):
    """获取任务状态"""
    if task := task_mgr.tasks.get(task_id):
        return response.json({
            "status": task["status"],
            "progress": f"{task['progress']:.1f}%",
            "error": task["error"]
        })
    return response.json({"error": "无效的任务ID"}, status=404)

@app.get("/task/result/<task_id>")
async def download_result(request, task_id):
    """下载结果（保持原有输出格式）"""
    if task := task_mgr.tasks.get(task_id):
        if task["status"] != "completed":
            return response.json({"error": "分析未完成"}, status=400)
        
        if Path(task["output"]).exists():
            return await response.file(
                task["output"],
                filename="comment_emotion.json",
                mime_type="application/json"
            )
    return response.json({"error": "结果文件不存在"}, status=404)

@app.listener("after_server_stop")
async def cleanup(app, loop):
    """清理临时文件"""
    shutil.rmtree(app.config.TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=2)
