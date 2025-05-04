from sanic import Sanic
from sanic.response import json as res_json, file
from concurrent.futures import ThreadPoolExecutor
import time
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import parse
import matplotlib

matplotlib.use('Agg')  # 非交互式后端

app = Sanic("SentimentAnalysisAPI")


def configure_logging():
    """统一日志配置（控制台+文件）"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 通用日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ========== 控制台处理器 ==========
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_level = logging.DEBUG if os.getenv("DEBUG") else logging.INFO
    console_handler.setLevel(console_level)

    # ========== 文件处理器 ==========
    # 应用日志
    app_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'application.log'),
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    app_handler.setFormatter(formatter)
    app_handler.setLevel(logging.INFO)

    # 配置根日志器（所有模块继承）
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_handler)
    root_logger.setLevel(logging.DEBUG)

    # Sanic专用日志配置
    # 访问日志
    access_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'access.log'),
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    access_handler.setFormatter(formatter)
    logging.getLogger("sanic.access").addHandler(access_handler)

    # 错误日志
    error_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'error.log'),
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    error_handler.setFormatter(formatter)
    logging.getLogger("sanic.error").addHandler(error_handler)


configure_logging()


@app.before_server_start
async def init_resources(app, _):
    """初始化资源"""
    try:
        app.ctx.executor = ThreadPoolExecutor(max_workers=4)
        app.ctx.logger = logging.getLogger("SA-Processor")
        app.ctx.logger.info("🔄 初始化线程池（4 workers）")
    except Exception as e:
        logging.error(f"资源初始化失败: {str(e)}", exc_info=True)
        raise


@app.after_server_stop
async def cleanup_resources(app, _):
    """清理资源"""
    try:
        app.ctx.executor.shutdown(wait=True)
        app.ctx.logger.info("🛑 线程池已关闭")
    except Exception as e:
        logging.error(f"资源清理失败: {str(e)}", exc_info=True)


os.makedirs("upload", exist_ok=True)
os.makedirs("sentiment-analysis", exist_ok=True)


def background_task(task_id):
    """后台分析任务"""
    task_logger = logging.getLogger(f"SA-Processor.Task.{task_id}")
    try:
        task_logger.info(f"🔍 开始处理任务 {task_id}")
        analyser = parse.Comment_analyser("upload", "sentiment-analysis", task_id)
        analyser.make_analyse()
        task_logger.info(f"✅ 任务完成 {task_id}")
    except Exception as e:
        task_logger.error(f"❌ 任务失败 {task_id}: {str(e)}", exc_info=True)


@app.route("/v1/movie/crawled/upload", methods=['POST'])
async def handle_upload(request):
    """文件上传接口"""
    logger = app.ctx.logger
    try:
        # 文件验证
        upload_file = request.files.get('file')
        if not upload_file:
            logger.warning("收到无文件请求")
            return res_json({"code": 0, "message": "文件未上传"}, ensure_ascii=False)

        # 文件名处理
        filename = upload_file.name
        if not filename.lower().endswith('.json'):
            logger.warning(f"非法文件类型: {filename}")
            return res_json({"code": 0, "message": "仅支持JSON文件"}, ensure_ascii=False)

        # 生成任务ID
        timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        task_id = f"{timestamp}_{os.path.splitext(filename)[0]}"
        save_path = os.path.join("upload", f"{task_id}.json")

        # 保存文件
        with open(save_path, 'wb') as f:
            f.write(upload_file.body)
        logger.info(f"📥 文件保存成功: {save_path}")

        # 提交后台任务
        app.ctx.executor.submit(background_task, task_id)
        logger.info(f"🚀 任务已提交: {task_id}")

        return res_json({
            "code": 1,
            "msg": "上传成功",
            "data": {
                "task_id": task_id,
                "download_url": f"/v1/image/download?filename={task_id}.zip"
            }
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"上传处理异常: {str(e)}", exc_info=True)
        return res_json({"code": 0, "msg": "服务器错误"}, ensure_ascii=False)


@app.route("/v1/image/download", methods=['GET'])
async def handle_download(request):
    """文件下载接口"""
    logger = app.ctx.logger
    try:
        filename = request.args.get("filename")
        if not filename:
            logger.warning("非法下载请求: 缺少文件名")
            return res_json({"code": 0, "msg": "缺少filename参数"}, ensure_ascii=False)

        # 安全验证
        if not filename.endswith('.zip') or '/' in filename:
            logger.warning(f"潜在路径遍历攻击: {filename}")
            return res_json({"code": 0, "msg": "非法文件请求"}, ensure_ascii=False)

        file_path = os.path.abspath(os.path.join("sentiment-analysis", filename))
        if not file_path.startswith(os.path.abspath("sentiment-analysis")):
            logger.warning(f"路径越界尝试: {filename}")
            return res_json({"code": 0, "msg": "非法文件路径"}, ensure_ascii=False)

        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {filename}")
            return res_json({"code": 0, "msg": "文件未找到"}, ensure_ascii=False)

        logger.info(f"📤 开始下载: {filename}")
        return await file(
            file_path,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"下载处理异常: {str(e)}", exc_info=True)
        return res_json({"code": 0, "msg": "下载错误"}, ensure_ascii=False)


@app.exception(Exception)
async def global_handler(request, exception):
    """全局异常处理"""
    logging.error(
        "未捕获异常",
        exc_info=exception,
        extra={
            "path": request.path,
            "method": request.method,
            "ip": request.remote_addr
        }
    )
    return res_json({
        "code": 0,
        "msg": "服务器内部错误",
        "error": str(exception)
    }, ensure_ascii=False)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8000,
        workers=4,
        debug=os.getenv("DEBUG", "false").lower() == "true",
        access_log=False
    )