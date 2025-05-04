from sanic import Sanic
from sanic.response import json as res_json, file
import time
import os
import matplotlib

matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor
import parse
import logging
from logging.handlers import TimedRotatingFileHandler

# 初始化Sanic应用
app = Sanic("mySanic")


# 配置日志系统
def setup_logging():
    """统一日志配置"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 标准格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Sanic专用日志器
    sanic_logger = logging.getLogger("mySanic")
    sanic_logger.setLevel(logging.INFO)

    # 按天切割的日志文件
    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'sanic.log'),
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    sanic_logger.addHandler(file_handler)
    app.logger = sanic_logger  # 手动绑定到app


# 执行日志配置
setup_logging()

@app.before_server_start
async def init_executor(app, loop):
    app.ctx.executor = ThreadPoolExecutor(max_workers=4)
    app.logger.info("线程池执行器初始化完成")

@app.after_server_stop
async def shutdown_executor(app, loop):
    app.ctx.executor.shutdown(wait=True)
    app.logger.info("线程池执行器已关闭")

# 确保目录存在
os.makedirs("upload", exist_ok=True)
os.makedirs("sentiment-analysis", exist_ok=True)

def analyser(id):
    try:
        in_path = os.path.join("upload")
        out_path = os.path.join("sentiment-analysis")
        comment_analyser = parse.Comment_analyser(in_path, out_path, id)
        comment_analyser.make_analyse()
        logging.info(f"情感分析完成 ID: {id}")
    except Exception as e:
        logging.error(f"影评分析失败 ID: {id}", exc_info=True)

@app.route("/v1/movie/crawled/upload", methods=['POST'])
async def upload(request):
    try:
        file = request.files.get('file')
        if not file:
            app.logger.warning("文件上传接口收到无文件请求")
            return res_json({"code": 0, "message": "文件未上传"}, ensure_ascii=False)

        filename = file.name
        _, ext = os.path.splitext(filename)
        if ext.lower() != '.json':
            app.logger.warning(f"文件格式错误: {filename}")
            return res_json({"code": 0, "message": "文件格式错误"}, ensure_ascii=False)

        now_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
        id = now_time + "_" + os.path.splitext(filename)[0]
        filename = id + ".json"
        path = os.path.join("upload", filename)

        with open(path, 'wb') as f:
            f.write(file.body)
        app.logger.info(f"文件保存成功: {path}")

        app.ctx.executor.submit(analyser, id)
        app.logger.info(f"分析任务已提交 ID: {id}")

        download_link = f"/v1/image/download?filename={id}.zip"

        return res_json({
            "code": 1,
            "msg": "上传成功，分析任务已提交",
            "data": {
                "name": filename,
                "download_url": download_link
            }
        }, ensure_ascii=False)

    except Exception as e:
        app.logger.error(f"文件上传接口异常: {str(e)}", exc_info=True)
        return res_json({"code": 0, "msg": f"服务器错误: {str(e)}"}, ensure_ascii=False)

@app.route("/v1/image/download", methods=['GET'])
async def download_image(request):
    try:
        filename = request.args.get("filename")
        if not filename:
            app.logger.warning("下载接口收到无文件名请求")
            return res_json({"code": 0, "msg": "缺少 filename 参数"}, ensure_ascii=False)

        safe_path = os.path.abspath(os.path.join("sentiment-analysis", filename))
        if not safe_path.startswith(os.path.abspath("sentiment-analysis")):
            app.logger.warning(f"非法路径访问尝试: {filename}")
            return res_json({"code": 0, "msg": "非法文件路径"}, ensure_ascii=False)

        if not os.path.exists(safe_path):
            app.logger.warning(f"文件不存在: {filename}")
            return res_json({"code": 0, "msg": "文件不存在"}, ensure_ascii=False)

        app.logger.info(f"开始文件下载: {filename}")
        return await file(
            safe_path,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        app.logger.error(f"文件下载接口异常: {str(e)}", exc_info=True)
        return res_json({"code": 0, "msg": f"服务器错误: {str(e)}"}, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, workers=4, debug=False, access_log=True)