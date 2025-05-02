import json
import uuid
import os
import shutil
import asyncio
import logging
import torch
from pathlib import Path
from sanic import Sanic, response
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

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
    "PORT": 8000
}

def create_app():
    """应用工厂函数"""
    app = Sanic("SentimentAnalysisCloud")
    app.config.update(CONFIG)
    
    # 初始化基础组件
    _configure_logging(app)
    _configure_device()
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

def _configure_device():
    """配置计算设备"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(CONFIG["GPU_MEM_FRACTION"])
        torch.cuda.empty_cache()
        logging.info(f"GPU initialized with {CONFIG['GPU_MEM_FRACTION']*100}% memory limit")
    else:
        logging.info("Using CPU for computation")

def _register_extensions(app):
    """注册扩展组件"""
    
    @app.after_server_start
    async def setup_model(app, loop):
        """延迟加载模型"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            app.ctx.model = pipeline(
                Tasks.text_classification,
                model='iic/nlp_structbert_sentiment-classification_chinese-base',
                device=device
            )
            app.ctx.task_mgr = TaskManager(app)
            logging.info(f"Model loaded on {device}")
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            app.stop()

class TaskManager:
    """任务管理系统"""
    
    def __init__(self, app):
        self.app = app
        self.tasks = {}
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=app.config.WORKERS)
        self._init_storage()
        
        # 启动任务处理器
        app.add_task(self.process_queue())
        app.add_task(self.cleanup_worker())
        
    def _init_storage(self):
        """初始化存储目录"""
        Path(self.app.config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        logging.info(f"Upload directory initialized at {self.app.config.UPLOAD_DIR}")

    async def process_queue(self):
        """任务队列处理器"""
        while True:
            task_id = await self.queue.get()
            async with self.lock:
                task = self.tasks[task_id]
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
                    task["status"] = "error"
                    task["error"] = str(e)
                logging.error(f"Task {task_id} failed: {str(e)}")
            finally:
                self.queue.task_done()

    async def _process_task(self, task_id):
        """单个任务处理流程"""
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
                        task["progress"] = round(progress, 1)
                
                outfile.write("\n]")

            # 阶段3：标记完成
            async with self.lock:
                task.update(
                    status="completed",
                    progress=100.0,
                    end_time=datetime.now().isoformat()
                )
            logging.info(f"Task {task_id} completed")
        
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
        try:
            inputs = self.app.ctx.model.preprocessor(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.app.ctx.model.device)
            
            with torch.no_grad():
                outputs = self.app.ctx.model.model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            return [
                {
                    "is_positive": int(prob[0] >= 0.5),
                    "positive_probs": float(prob[0]),
                    "negative_probs": float(prob[1])
                } for prob in probs
            ]
        except Exception as e:
            logging.error(f"Batch processing failed: {str(e)}")
            return [None] * len(batch)

    async def _write_batch(self, results, outfile, processed, total):
        """写入批次结果"""
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return
        
        json_str = ",\n".join(json.dumps(r) for r in valid_results)
        if processed + len(valid_results) < total:
            json_str += ","
        
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: outfile.write(json_str + "\n")
        )

    async def cleanup_worker(self):
        """定时清理旧任务"""
        while True:
            await asyncio.sleep(3600)  # 每小时清理一次
            cutoff = datetime.now() - timedelta(hours=CONFIG["TASK_RETENTION"])
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

def _register_routes(app):
    """注册路由端点"""
    
    @app.post("/api/v1/analyze")
    async def analyze(request):
        """文件上传接口"""
        if not (upload_file := request.files.get("file")):
            return response.json({"error": "No file provided"}, status=400)
        
        if not upload_file.name.lower().endswith(".jsonl"):
            return response.json({"error": "Invalid file type"}, status=400)
        
        if len(upload_file.body) > app.config.MAX_FILE_SIZE:
            return response.json({"error": "File size exceeds limit"}, status=413)
        
        # 创建任务记录
        task_id = str(uuid.uuid4())
        input_path = Path(app.config.UPLOAD_DIR) / f"{task_id}.jsonl"
        output_path = Path(app.config.UPLOAD_DIR) / f"{task_id}.json"
        
        try:
            with open(input_path, "wb") as f:
                f.write(upload_file.body)
            
            async with app.ctx.task_mgr.lock:
                app.ctx.task_mgr.tasks[task_id] = {
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
            
            await app.ctx.task_mgr.queue.put(task_id)
            return response.json({
                "task_id": task_id,
                "status": f"/api/v1/tasks/{task_id}",
                "download": f"/api/v1/results/{task_id}"
            })
        
        except Exception as e:
            logging.error(f"Upload failed: {str(e)}")
            return response.json({"error": "Internal server error"}, status=500)

    @app.get("/api/v1/tasks/<task_id:str>")
    async def get_task(request, task_id: str):
        """任务状态查询"""
        async with app.ctx.task_mgr.lock:
            task = app.ctx.task_mgr.tasks.get(task_id)
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
        """结果下载接口"""
        async with app.ctx.task_mgr.lock:
            task = app.ctx.task_mgr.tasks.get(task_id)
            if not task:
                return response.json({"error": "Task not found"}, status=404)
            
            if task["status"] != "completed":
                return response.json({"error": "Task not completed"}, status=400)
            
            output_path = Path(task["output_path"])
            if not output_path.exists():
                return response.json({"error": "Result file missing"}, status=404)
            
            return await response.file(
                output_path,
                filename=f"analysis_{task_id}.json",
                mime_type="application/json"
            )

    @app.get("/api/v1/health")
    async def health_check(request):
        """健康检查端点"""
        return response.json({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "gpu_available": torch.cuda.is_available(),
            "active_tasks": len(app.ctx.task_mgr.tasks),
            "system_load": os.getloadavg(),
            "memory_usage": f"{os.getpid().memory_info().rss/1024/1024:.2f}MB"
        })

def _register_cleanup(app):
    """注册清理钩子"""
    
    @app.after_server_stop
    async def cleanup_system(app, _):
        """系统清理"""
        logging.info("Starting system cleanup...")
        app.ctx.task_mgr.executor.shutdown()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
