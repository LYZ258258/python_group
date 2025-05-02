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

app = Sanic("SentimentAnalysisService")
app.config.update({
    "UPLOAD_DIR": "/data/uploads",
    "MAX_FILE_SIZE": 10 * 1024 * 1024,  # 10MB
    "MAX_CONCURRENT": 2,                 # 最大并行任务数
    "BATCH_SIZE": 64,                    # 批处理大小
    "MODEL_WORKERS": 1                   # 模型线程数
})

# 初始化模型
model = pipeline(
    Tasks.text_classification,
    'iic/nlp_structbert_sentiment-classification_chinese-base',
    device='cpu'
)

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=app.config.MAX_CONCURRENT)
        Path(app.config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    async def process_queue(self):
        """任务处理队列"""
        while True:
            task_id = await self.queue.get()
            async with self.lock:
                task = self.tasks[task_id]
                task["status"] = "processing"
            
            try:
                await self._process_task(task_id)
            except Exception as e:
                async with self.lock:
                    task["status"] = "error"
                    task["error"] = str(e)
            finally:
                self.queue.task_done()

    async def _process_task(self, task_id):
        """处理单个任务"""
        task = self.tasks[task_id]
        input_path = Path(task["input_path"])
        output_path = Path(task["output_path"])
        
        try:
            # 统计总行数
            total = 0
            with open(input_path, "r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
            
            # 批处理
            batch = []
            current = 0
            with open(input_path, "r", encoding="utf-8") as infile, \
                 open(output_path, "w", encoding="utf-8") as outfile:
                
                outfile.write("[\n")
                
                for line in infile:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    if content := data.get("content", "").strip():
                        batch.append(content)
                    
                    # 处理批次
                    if len(batch) >= app.config.BATCH_SIZE:
                        current = await self._process_batch(batch, current, total, outfile)
                        batch = []
                
                # 处理剩余
                if batch:
                    current = await self._process_batch(batch, current, total, outfile)
                
                outfile.write("\n]")
            
            async with self.lock:
                task.update(status="completed", progress=100)
        
        finally:
            input_path.unlink(missing_ok=True)

    async def _process_batch(self, batch, current, total, outfile):
        """处理数据批次"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            lambda: [model(text) for text in batch]
        )
        
        records = []
        for text, res in zip(batch, results):
            labels = res["labels"]
            scores = res["scores"]
            records.append({
                "is_positive": int(scores[0] >= 0.5),
                "positive_probs": scores[0],
                "negative_probs": scores[1]
            })
        
        # 写入文件
        outfile.write(",\n".join(json.dumps(r) for r in records))
        if current + len(batch) < total:
            outfile.write(",")
        
        # 更新进度
        new_current = current + len(batch)
        progress = min(99.9, (new_current / total) * 100)
        async with self.lock:
            self.tasks[task["id"]]["progress"] = round(progress, 1)
        
        return new_current

task_mgr = TaskManager()

@app.listener('after_server_start')
async def init(app, loop):
    app.add_task(task_mgr.process_queue())

@app.post("/analyze")
async def analyze(request):
    """文件上传接口"""
    # 验证文件
    if not (upload_file := request.files.get('file')):
        return response.json({"error": "请上传文件"}, status=400)
    
    if not upload_file.name.lower().endswith('.jsonl'):
        return response.json({"error": "仅支持.jsonl文件"}, status=400)
    
    if len(upload_file.body) > app.config.MAX_FILE_SIZE:
        return response.json({"error": "文件超过10MB限制"}, status=413)
    
    # 保存文件
    task_id = str(uuid.uuid4())
    input_path = Path(app.config.UPLOAD_DIR) / f"{task_id}.jsonl"
    output_path = Path(app.config.UPLOAD_DIR) / f"{task_id}.json"
    
    with open(input_path, "wb") as f:
        f.write(upload_file.body)
    
    # 创建任务
    async with task_mgr.lock:
        task_mgr.tasks[task_id] = {
            "id": task_id,
            "status": "queued",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "progress": 0.0,
            "error": None
        }
    
    await task_mgr.queue.put(task_id)
    return response.json({
        "task_id": task_id,
        "status": f"/task/{task_id}",
        "download": f"/download/{task_id}"
    })

@app.get("/task/<task_id>")
async def get_task(task_id):
    """获取任务状态"""
    async with task_mgr.lock:
        if task_id not in task_mgr.tasks:
            return response.json({"error": "任务不存在"}, status=404)
        
        task = task_mgr.tasks[task_id]
        return response.json({
            "status": task["status"],
            "progress": task["progress"],
            "error": task["error"]
        })

@app.get("/download/<task_id>")
async def download(task_id):
    """下载结果"""
    async with task_mgr.lock:
        if task_id not in task_mgr.tasks:
            return response.json({"error": "任务不存在"}, status=404)
        
        task = task_mgr.tasks[task_id]
        if task["status"] != "completed":
            return response.json({"error": "分析未完成"}, status=400)
        
        if not Path(task["output_path"]).exists():
            return response.json({"error": "结果文件不存在"}, status=404)
        
        return await response.file(
            task["output_path"],
            filename="result.json",
            mime_type="application/json"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, access_log=False)
