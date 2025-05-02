import json
import uuid
import os
import shutil
import asyncio
import logging
import torch
from pathlib import Path
from sanic import Sanic, response
from sanic.worker.manager import WorkerManager
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# 禁用Sanic自动重载
WorkerManager.THRESHOLD = 86400  # 24小时

# 全局配置项
CONFIG = {
    "UPLOAD_DIR": "/data/uploads",
    "MAX_FILE_SIZE": 10 * 1024 * 1024,  # 10MB
    "WORKERS": 1,                        # 根据GPU数量调整
    "BATCH_SIZE": 64,                    # 处理批次大小
    "TASK_TIMEOUT": 3600,                # 任务超时时间(秒)
    "LOG_LEVEL": logging.INFO,           # 日志级别
    "GPU_MEM_FRACTION": 0.5,             # GPU显存占比(每个进程)
    "TASK_RETENTION": 24,                # 任务保留时间(小时)
    "HOST": "0.0.0.0",
    "PORT": 8000,
    "MODEL_NAME": "iic/nlp_structbert_sentiment-classification_chinese-base"
}

def create_app():
    """应用工厂函数"""
    app = Sanic("SentimentAnalysisCloud")
    app.config.update(CONFIG)
    
    # 初始化阶段
    _configure_logging(app)
    _configure_system(app)
    _register_extensions(app)
    _register_routes(app)
    _register_cleanup(app)
    
    return app

def _configure_logging(app):
    """配置日志系统"""
    logging.basicConfig(
        level=app.config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("service.log"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("sanic").setLevel(logging.WARNING)

def _configure_system(app):
    """系统级配置"""
    # 创建上传目录
    Path(app.config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    
    # 配置GPU显存
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(app.config.GPU_MEM_FRACTION)
        torch.cuda.empty_cache()
        logging.info(f"Initialized GPU with {app.config.GPU_MEM_FRACTION*100}% memory limit")

class TaskManager:
    """完整任务管理系统"""
    
    def __init__(self, app):
        self.app = app
        self.tasks = {}
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=app.config.WORKERS)
        self.model = None
        self._ready = False

    async def initialize(self):
        """延迟初始化模型"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = pipeline(
                Tasks.text_classification,
                model=self.app.config.MODEL_NAME,
                device=device
            )
            logging.info(f"Model loaded on {device}")
            
            # 启动后台任务
            self.app.add_task(self._process_queue())
            self.app.add_task(self._cleanup_worker())
            self._ready = True
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            self.app.stop()

    async def _process_queue(self):
        """处理任务队列"""
        while True:
            task_id = await self.queue.get()
            async with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    continue
                task.update(
                    status="processing",
                    start_time=datetime.now().isoformat()
                )
            
            try:
                await asyncio.wait_for(
                    self._process_task(task_id),
                    timeout=self.app.config.TASK_TIMEOUT
                )
            except Exception as e:
                async with self.lock:
                    if task_id in self.tasks:
                        self.tasks[task_id].update(
                            status="error",
                            error=str(e)
                        )
                logging.error(f"Task {task_id} failed: {str(e)}")
            finally:
                self.queue.task_done()

    async def _process_task(self, task_id):
        """处理单个任务"""
        task = self.tasks[task_id]
        input_path = Path(task["input_path"])
        output_path = Path(task["output_path"])
        
        try:
            # 阶段1：准备数据
            total = await self._count_valid_lines(input_path)
            if total == 0:
                raise ValueError("No valid content found")
            
            # 阶段2：批处理
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write("[\n")
                processed = 0
                
                async for batch in self._generate_batches(input_path):
                    results = await self._process_batch(batch)
                    await self._write_batch(results, outfile, processed, total)
                    processed += len(batch)
                    
                    # 更新进度
                    progress = min(99.9, processed / total * 100)
                    async with self.lock:
                        if task_id in self.tasks:
                            self.tasks[task_id]["progress"] = round(progress, 1)
                
                outfile.write("\n]")

            # 阶段3：标记完成
            async with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update(
                        status="completed",
                        progress=100.0,
                        end_time=datetime.now().isoformat()
                    )
        finally:
            input_path.unlink(missing_ok=True)

    async def _count_valid_lines(self, file_path):
        """统计有效行数"""
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    async def _generate_batches(self, file_path):
        """生成数据批次"""
        batch = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    if content := data.get("content", "").strip():
                        batch.append(content)
                        
                        if len(batch) == self.app.config.BATCH_SIZE:
                            yield batch
                            batch = []
                except json.JSONDecodeError:
                    continue
            
            if batch:
                yield batch

    async def _process_batch(self, batch):
        """处理单批次数据"""
        inputs = self.model.preprocessor(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return [
            {
                "is_positive": int(prob[0] >= 0.5),
                "positive_probs": float(prob[0]),
                "negative_probs": float(prob[1])
            } for prob in probs
        ]

    async def _write_batch(self, results, outfile, processed, total):
        """写入批次结果"""
        if not results:
            return
        
        json_str = ",\n".join(json.dumps(r) for r in results)
        if processed + len(results) < total:
            json_str += ","
        
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: outfile.write(json_str + "\n")
        )

    async def _cleanup_worker(self):
        """清理旧任务"""
        while True:
            await asyncio.sleep(3600)
            cutoff = datetime.now() - timedelta(hours=self.app.config.TASK_RETENTION)
            async with self.lock:
                expired = [
                    tid for tid, task in self.tasks.items()
                    if datetime.fromisoformat(task["created"]) < cutoff
                ]
                for tid in expired:
                    output_path = Path(self.tasks[tid]["output_path"])
                    output_path.unlink(missing_ok=True)
                    del self.tasks[tid]
                logging.info(f"Cleaned up {len(expired)} expired tasks")

def _register_extensions(app):
    """注册扩展组件"""
    app.ctx.task_mgr = TaskManager(app)
    
    @app.after_server_start
    async def setup_system(app, loop):
        """系统初始化"""
        await app.ctx.task_mgr.initialize()

def _register_routes(app):
    """注册API路由"""
    
    @app.post("/api/v1/analyze")
    async def analyze(request):
        """文件上传接口"""
        if not request.app.ctx.task_mgr._ready:
            return response.json({"error": "System initializing"}, status=503)
        
        # 验证文件
        if not (upload_file := request.files.get("file")):
            return response.json({"error": "No file provided"}, status=400)
        
        if not upload_file.name.lower().endswith(".jsonl"):
            return response.json({"error": "Invalid file type"}, status=400)
        
        if len(upload_file.body) > request.app.config.MAX_FILE_SIZE:
            return response.json({"error": "File size exceeds limit"}, status=413)
        
        # 创建任务记录
        task_id = str(uuid.uuid4())
        input_path = Path(app.config.UPLOAD_DIR) / f"{task_id}.jsonl"
        output_path = Path(app.config.UPLOAD_DIR) / f"{task_id}.json"
        
        try:
            # 保存上传文件
            with open(input_path, "wb") as f:
                f.write(upload_file.body)
            
            # 记录任务信息
            async with request.app.ctx.task_mgr.lock:
                request.app.ctx.task_mgr.tasks[task_id] = {
                    "id": task_id,
                    "status": "queued",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "progress": 0.0,
                    "error": None,
                    "created": datetime.now().isoformat(),
                    "start_time": None,
                    "end_time": None
                }
            
            # 提交任务到队列
            await request.app.ctx.task_mgr.queue.put(task_id)
            return response.json({
                "task_id": task_id,
                "status_url": f"/api/v1/tasks/{task_id}",
                "download_url": f"/api/v1/results/{task_id}"
            })
        
        except Exception as e:
            logging.error(f"Upload failed: {str(e)}")
            return response.json({"error": "Internal server error"}, status=500)

    @app.get("/api/v1/tasks/<task_id:str>")
    async def get_task(request, task_id: str):
        """查询任务状态"""
        async with request.app.ctx.task_mgr.lock:
            task = request.app.ctx.task_mgr.tasks.get(task_id)
            if not task:
                return response.json({"error": "Task not found"}, status=404)
            
            return response.json({
                "id": task["id"],
                "status": task["status"],
                "progress": task["progress"],
                "created": task["created"],
                "start_time": task["start_time"],
                "end_time": task["end_time"],
                "error": task["error"]
            })

    @app.get("/api/v1/results/<task_id:str>")
    async def download_result(request, task_id: str):
        """下载分析结果"""
        async with request.app.ctx.task_mgr.lock:
            task = request.app.ctx.task_mgr.tasks.get(task_id)
            if not task:
                return response.json({"error": "Task not found"}, status=404)
            
            if task["status"] != "completed":
                return response.json({"error": "Analysis not completed"}, status=400)
            
            output_path = Path(task["output_path"])
            if not output_path.exists():
                return response.json({"error": "Result file missing"}, status=404)
            
            return await response.file(
                output_path,
                filename=f"sentiment_analysis_{task_id}.json",
                mime_type="application/json"
            )

    @app.get("/api/v1/health")
    async def health_check(request):
        """健康检查接口"""
        status = {
            "status": "ready" if request.app.ctx.task_mgr._ready else "initializing",
            "timestamp": datetime.now().isoformat(),
            "active_tasks": len(request.app.ctx.task_mgr.tasks),
            "gpu_available": torch.cuda.is_available(),
            "system_load": os.getloadavg(),
            "memory_usage": f"{os.getpid().memory_info().rss / 1024 / 1024:.2f}MB"
        }
        return response.json(status)

def _register_cleanup(app):
    """注册清理钩子"""
    
    @app.after_server_stop
    async def cleanup_system(app, _):
        """系统关闭清理"""
        logging.info("Starting system cleanup...")
        
        # 关闭线程池
        app.ctx.task_mgr.executor.shutdown()
        
        # 清理GPU资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 删除临时文件
        shutil.rmtree(app.config.UPLOAD_DIR, ignore_errors=True)
        
        logging.info("System cleanup completed")

if __name__ == "__main__":
    app = create_app()
    app.run(
        host=CONFIG["HOST"],
        port=CONFIG["PORT"],
        workers=CONFIG["WORKERS"],
        single_process=(CONFIG["WORKERS"] == 1),
        access_log=False,
        motd=False,
        auto_reload=False
    )
