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

# 服务配置
app = Sanic("SentimentAnalysisCloud")
app.config.update({
    "UPLOAD_DIR": "/data/uploads",
    "MAX_FILE_SIZE": 10 * 1024 * 1024,  # 10MB
    "MAX_CONCURRENT_TASKS": 2,          # 并行任务数=CPU核心数
    "BATCH_SIZE": 64,                   # 根据内存调整(64-256)
    "TASK_TIMEOUT": 3600,               # 任务超时(秒)
    "LOG_LEVEL": logging.INFO
})

# 初始化日志
logging.basicConfig(
    level=app.config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设备初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 模型初始化
try:
    nlp_pipeline = pipeline(
        task=Tasks.text_classification,
        model='iic/nlp_structbert_sentiment-classification_chinese-base'
    )
    nlp_pipeline.model = nlp_pipeline.model.to(device)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=app.config.MAX_CONCURRENT_TASKS)
        self._init_upload_dir()

    def _init_upload_dir(self):
        Path(app.config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Upload directory initialized at {app.config.UPLOAD_DIR}")

    async def task_scheduler(self):
        """任务调度器"""
        while True:
            task_id = await self.queue.get()
            async with self.lock:
                task = self.tasks[task_id]
                task["status"] = "processing"
                task["start_time"] = datetime.now().isoformat()
            
            try:
                await asyncio.wait_for(
                    self._process_task(task_id),
                    timeout=app.config.TASK_TIMEOUT
                )
            except Exception as e:
                async with self.lock:
                    task["status"] = "error"
                    task["error"] = str(e)
                logger.error(f"Task {task_id} failed: {str(e)}")
            finally:
                self.queue.task_done()

    async def _process_task(self, task_id):
        """任务处理核心逻辑"""
        task = self.tasks[task_id]
        input_path = Path(task["input_path"])
        output_path = Path(task["output_path"])
        
        try:
            # 阶段1: 准备数据
            total = await self._count_lines(input_path)
            if total == 0:
                raise ValueError("No valid content found")
            
            # 阶段2: 批处理
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

            # 阶段3: 完成处理
            async with self.lock:
                task.update(
                    status="completed",
                    progress=100.0,
                    end_time=datetime.now().isoformat()
                )
            logger.info(f"Task {task_id} completed")
        
        finally:
            # 清理资源
            input_path.unlink(missing_ok=True)

    async def _count_lines(self, input_path):
        """统计有效行数"""
        count = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    async def _generate_batches(self, input_path):
        """生成数据批次"""
        batch = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    if content := data.get("content", "").strip():
                        batch.append(content)
                        
                        if len(batch) == app.config.BATCH_SIZE:
                            yield batch
                            batch = []
                except json.JSONDecodeError:
                    continue
            
            if batch:
                yield batch

    async def _process_batch(self, batch):
        """处理单批次数据"""
        loop = asyncio.get_event_loop()
        try:
            inputs = nlp_pipeline.preprocessor(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = nlp_pipeline.model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            return [
                {
                    "is_positive": int(prob[0] >= 0.5),
                    "positive_probs": float(prob[0]),
                    "negative_probs": float(prob[1])
                } for prob in probs
            ]
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
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

task_mgr = TaskManager()

@app.listener('after_server_start')
async def init_system(app, loop):
    app.add_task(task_mgr.task_scheduler())
    app.add_task(_cleanup_worker())
    logger.info("System initialized")

async def _cleanup_worker():
    """定时清理旧任务"""
    while True:
        await asyncio.sleep(3600)  # 每小时清理一次
        now = datetime.now()
        expired = []
        
        async with task_mgr.lock:
            for tid, task in task_mgr.tasks.items():
                create_time = datetime.fromisoformat(task["created"])
                if now - create_time > timedelta(hours=24):
                    expired.append(tid)
            
            for tid in expired:
                if tid in task_mgr.tasks:  # 添加存在性检查
                    output_path = Path(task_mgr.tasks[tid]["output_path"])
                    output_path.unlink(missing_ok=True)
                    del task_mgr.tasks[tid]
        
        logger.info(f"Cleaned up {len(expired)} expired tasks")

# 文件上传、状态查询和下载接口保持不变（与之前相同）
# ...

@app.listener('before_server_stop')
async def shutdown(app, loop):
    """服务关闭时清理资源"""
    logger.info("Shutting down system...")
    # 清理模型显存
    if torch.cuda.is_available():
        nlp_pipeline.model.cpu()
        torch.cuda.empty_cache()
    # 关闭线程池
    task_mgr.executor.shutdown()
    logger.info("System shutdown complete")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        access_log=False,
        motd=False,
        auto_reload=False
    )
