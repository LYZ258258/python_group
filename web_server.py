# web_server.py
from sanic import Sanic
from sanic.response import json as res_json, file
from concurrent.futures import ThreadPoolExecutor
import time
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import parse
import matplotlib

matplotlib.use('Agg')  # 必须在导入pyplot前设置

# 初始化Sanic应用
app = Sanic("SentimentAnalysisAPI")


# 日志配置函数
def configure_logging():
    """配置日志系统（文件+控制台）"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 通用日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ================= 控制台处理器 =================
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(
        logging.DEBUG if os.getenv("DEBUG") else logging.INFO
    )

    # ================= 文件处理器 =================
    # 访问日志
    access_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'access.log'),
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    access_handler.setFormatter(formatter)
    access_logger = logging.getLogger("sanic.access")
    access_logger.addHandler(access_handler)
    access_logger.setLevel(logging.INFO)

    # 错误日志
    error_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'error.log'),
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    error_handler.setFormatter(formatter)
    error_logger = logging.getLogger("sanic.error")
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)

    # 应用业务日志
    app_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'application.log'),
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    app_handler.setFormatter(formatter)

    # ================= 应用日志器配置 =================
    app_logger = logging.getLogger("SA-Processor")
    app_logger.addHandler(app_handler)  # 文件输出
    app_logger.addHandler(console_handler)  # 控制台输出
    app_logger.setLevel(logging.DEBUG if os.getenv("DEBUG") else logging.INFO)

    # 存储到上下文
    app.ctx.logger = app_logger


# 执行日志配置
configure_logging()


@app.before_server_start
async def init_resources(app, _):
    """初始化线程池"""
    try:
        app.ctx.executor = ThreadPoolExecutor(max_workers=4)
        app.ctx.logger.info("🔄 初始化线程池（4 workers）")
    except Exception as e:
        app.ctx.logger.error(f"线程池初始化失败: {str(e)}", exc_info=True)
        raise


@app.after_server_stop
async def cleanup_resources(app, _):
    """清理资源"""
    try:
        app.ctx.executor.shutdown(wait=True)
        app.ctx.logger.info("🛑 线程池已关闭")
    except Exception as e:
        app.ctx.logger.error(f"资源清理失败: {str(e)}", exc_info=True)


# 确保目录存在
os.makedirs("upload", exist_ok=True)
os.makedirs("sentiment-analysis", exist_ok=True)


def background_analysis(task_id):
    """后台分析任务"""
    logger = app.ctx.logger.getChild(task_id)
    try:
        logger.info(f"🔍 开始分析任务 {task_id}")
        in_path = "upload"
        out_path = "sentiment-analysis"

        analyser = parse.Comment_analyser(in_path, out_path, task_id)
        analyser.make_analyse()

        logger.info(f"✅ 分析完成 {task_id}")
        return True
    except Exception as e:
        logger.error(f"❌ 分析失败 {task_id}: {str(e)}", exc_info=True)
        return False


@app.route("/v1/movie/crawled/upload", methods=['POST'])
async def handle_upload(request):
    """处理文件上传"""
    logger = app.ctx.logger
    try:
        # 验证文件
        upload_file = request.files.get('file')
        if not upload_file:
            logger.warning("收到无文件请求")
            return res_json({"code": 0, "message": "文件未上传"}, ensure_ascii=False)

        # 验证扩展名
        filename = upload_file.name
        if not filename.lower().endswith('.json'):
            logger.warning(f"非法文件类型: {filename}")
            return res_json({"code": 0, "message": "仅支持JSON文件"}, ensure_ascii=False)

        # 生成唯一ID
        timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        task_id = f"{timestamp}_{os.path.splitext(filename)[0]}"
        save_name = f"{task_id}.json"
        save_path = os.path.join("upload", save_name)

        # 保存文件
        with open(save_path, 'wb') as f:
            f.write(upload_file.body)
        logger.info(f"📥 文件保存成功: {save_path}")

        # 提交后台任务
        app.ctx.executor.submit(background_analysis, task_id)
        logger.info(f"🚀 提交分析任务: {task_id}")

        # 生成响应
        return res_json({
            "code": 1,
            "msg": "上传成功，分析已开始",
            "data": {
                "task_id": task_id,
                "download_url": f"/v1/image/download?filename={task_id}.zip"
            }
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"上传处理异常: {str(e)}", exc_info=True)
        return res_json({"code": 0, "msg": "服务器处理错误"}, status=500)


@app.route("/v1/image/download", methods=['GET'])
async def handle_download(request):
    """处理文件下载"""
    logger = app.ctx.logger
    try:
        filename = request.args.get("filename")
        if not filename:
            logger.warning("下载请求缺少文件名")
            return res_json({"code": 0, "msg": "缺少filename参数"}, status=400)

        # 安全路径验证
        if '../' in filename or not filename.endswith('.zip'):
            logger.warning(f"非法文件名: {filename}")
            return res_json({"code": 0, "msg": "非法文件请求"}, status=403)

        file_path = os.path.abspath(os.path.join("sentiment-analysis", filename))
        if not file_path.startswith(os.path.abspath("sentiment-analysis")):
            logger.warning(f"路径越界尝试: {filename}")
            return res_json({"code": 0, "msg": "非法文件路径"}, status=403)

        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {filename}")
            return res_json({"code": 0, "msg": "文件未找到"}, status=404)

        logger.info(f"📤 开始下载: {filename}")
        return await file(
            file_path,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except Exception as e:
        logger.error(f"下载处理异常: {str(e)}", exc_info=True)
        return res_json({"code": 0, "msg": "下载处理错误"}, status=500)


@app.exception(Exception)
async def global_exception_handler(request, exception):
    """全局异常处理"""
    app.ctx.logger.error(
        f"未捕获异常: {str(exception)}",
        exc_info=True,
        extra={"path": request.path, "method": request.method}
    )
    return res_json({
        "code": 0,
        "msg": "服务器内部错误",
        "error": str(exception)
    }, status=500)


if __name__ == '__main__':
    # 开发模式：DEBUG=1 python web_server.py
    app.run(
        host='0.0.0.0',
        port=8000,
        workers=4,
        debug=os.getenv("DEBUG", "").lower() == "true",  # 根据环境变量控制
        access_log=False  # 禁用Sanic内置访问日志（已自定义）
    )