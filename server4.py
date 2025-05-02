import json
import uuid
import os
import shutil
import asyncio
from pathlib import Path
from sanic import Sanic, response
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# 服务配置
app = Sanic("SentimentAnalysisCloud")
app.config.update({
    "TEMP_DIR": "/data/tmp_analysis",
    "MAX_FILE_SIZE": 10 * 1024 * 1024,  # 10MB
    "ALLOWED_EXTENSIONS": {"jsonl"},
    "MAX_CONCURRENT_TASKS": 2,          # 最大并行任务数
    "MODEL_WORKERS": 1,                 # 模型并行线程数
    "BATCH_SIZE": 64,                   # 批处理大小
    "TASK_TIMEOUT": 3600                # 任务超时时间(秒)
})

# 全局资源初始化
model = pipeline(
    Tasks.text_classification,
    'iic/nlp_structbert_sentiment-classification_chinese-base',
    device='cpu'
)
model_executor = ThreadPoolExecutor(max_workers=app.config.MODEL_WORKERS)

class AnalysisTask:
    def __init__(self):
        self.tasks = {}
        self.lock = asyncio.Lock()
        self.task_queue = asyncio.Queue()
        Path(app.config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    async def task_worker(self):
        """任务处理工作线程"""
        while True:
            task_id = await self.task_queue.get()
            async with self.lock:
                task = self.tasks[task_id]
                task["status"] = "processing"
                task["start_time"] = datetime.now().isoformat()
            
            try:
                await self.process_task(task_id)
            except Exception as e:
                async with self.lock:
                    task["status"] = "error"
                    task["error"] = str(e)
            finally:
                self.task_queue.task_done()

    async def create_task(self, file_path):
        """创建新任务"""
        task_id = str(uuid.uuid4())
        task_dir = Path(app.config.TEMP_DIR) / task_id
        task_dir.mkdir()

        async with self.lock:
            self.tasks[task_id] = {
                "id": task_id,
                "status": "queued",
                "input": str(task_dir / "input.jsonl"),
                "output": str(task_dir / "result.json"),
                "progress": 0.0,
                "error": None,
                "created": datetime.now().isoformat(),
                "start_time": None,
                "end_time": None
            }
        
        shutil.move(file_path, self.tasks[task_id]["input"])
        await self.task_queue.put(task_id)
        return task_id

task_mgr = AnalysisTask()

@app.listener('after_server_start')
async def init_workers(app, loop):
    for _ in range(app.config.MAX_CONCURRENT_TASKS):
        app.add_task(task_mgr.task_worker())

@app.post("/api/v1/analyze")
async def upload_file(request):
    """文件上传接口"""
    # 验证文件存在
    if not (upload_file := request.files.get('file')):
        return response.json({"error": "No file provided"}, status=400)
    
    # 验证文件类型
    if not _is_valid_file(upload_file.name):
        return response.json({"error": "Invalid file type"}, status=400)
    
    # 验证文件大小
    if len(upload_file.body) > app.config.MAX_FILE_SIZE:
        return response.json({"error": "File too large"}, status=413)

    # 保存临时文件
    temp_path = Path(app.config.TEMP_DIR) / f"upload_{uuid.uuid4().hex}.jsonl"
    with open(temp_path, "wb") as f:
        f.write(upload_file.body)

    # 创建任务
    try:
        task_id = await task_mgr.create_task(temp_path)
        return response.json({
            "task_id": task_id,
            "status_url": f"/api/v1/tasks/{task_id}",
            "download_url": f"/api/v1/results/{task_id}"
        })
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

async def process_task(task_id):
    """任务处理核心逻辑"""
    task = task_mgr.tasks[task_id]
    
    try:
        # 第一阶段：准备数据
        batch = []
        total = 0
        input_path = Path(task["input"])
        
        # 统计总数
        with open(input_path, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())
        
        # 第二阶段：批处理分析
        with open(input_path, "r", encoding="utf-8") as in_file, \
             open(task["output"], "w", encoding="utf-8") as out_file:
            
            out_file.write("[\n")
            first_item = True
            current_count = 0
            
            # 分批读取
            for line in in_file:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get("content", "").strip()
                    if text:
                        batch.append(text)
                except json.JSONDecodeError:
                    continue
                
                # 批量处理
                if len(batch) >= app.config.BATCH_SIZE:
                    await _process_batch(batch, out_file, task, current_count, total, first_item)
                    current_count += len(batch)
                    batch = []
                    first_item = False
            
            # 处理剩余批次
            if batch:
                await _process_batch(batch, out_file, task, current_count, total, first_item)
            
            out_file.write("\n]")

        # 更新任务状态
        async with task_mgr.lock:
            task["status"] = "completed"
            task["progress"] = 100.0
            task["end_time"] = datetime.now().isoformat()

    finally:
        # 清理输入文件
        if input_path.exists():
            input_path.unlink()

async def _process_batch(batch, out_file, task, current_count, total, first_item):
    """处理单个批次"""
    # 执行模型推理
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        model_executor,
        lambda: [model(text) for text in batch]
    )
    
    # 写入结果
    for i, (text, res) in enumerate(zip(batch, results)):
        labels = res["labels"]
        scores = res["scores"]
        
        record = {
            "is_positive": int(scores[labels.index("正面")] >= 0.5),
            "positive_probs": scores[labels.index("正面")],
            "negative_probs": scores[labels.index("负面")]
        }
        
        # 格式化输出
        prefix = ",\n" if not first_item or i > 0 else ""
        out_file.write(f"{prefix}{json.dumps(record, ensure_ascii=False)}")
    
    # 更新进度
    processed = current_count + len(batch)
    progress = min(99.9, processed / total * 100)
    async with task_mgr.lock:
        task["progress"] = round(progress, 1)

@app.get("/api/v1/tasks/<task_id>")
async def get_task_status(request, task_id):
    """获取任务状态"""
    async with task_mgr.lock:
        if task_id not in task_mgr.tasks:
            return response.json({"error": "Task not found"}, status=404)
        
        task = task_mgr.tasks[task_id]
        return response.json({
            "id": task["id"],
            "status": task["status"],
            "progress": task["progress"],
            "created": task["created"],
            "start_time": task["start_time"],
            "end_time": task["end_time"],
            "error": task["error"]
        })

@app.get("/api/v1/results/<task_id>")
async def download_result(request, task_id):
    """下载结果文件"""
    async with task_mgr.lock:
        if task_id not in task_mgr.tasks:
            return response.json({"error": "Task not found"}, status=404)
        
        task = task_mgr.tasks[task_id]
        if task["status"] != "completed":
            return response.json({"error": "Task not completed"}, status=400)
        
        output_path = Path(task["output"])
        if not output_path.exists():
            return response.json({"error": "Result file missing"}, status=500)
        
        return await response.file(
            output_path,
            filename="analysis_result.json",
            mime_type="application/json"
        )

@app.listener('after_server_stop')
async def cleanup(app, loop):
    """清理资源"""
    # 清理临时目录
    shutil.rmtree(app.config.TEMP_DIR, ignore_errors=True)
    # 关闭线程池
    model_executor.shutdown()

def _is_valid_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in app.config.ALLOWED_EXTENSIONS)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        access_log=False,
        motd=False,
        auto_reload=False
    )
