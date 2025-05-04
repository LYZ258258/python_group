from sanic import Sanic
from sanic.response import json as res_json, file
import time
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 线程冲突
from concurrent.futures import ThreadPoolExecutor
import asyncio
import parse
import netifaces

# 本机ip列表
ips = []
# 端口号
port=8000

app = Sanic("mySanic")
executor = None

@app.before_server_start
async def init_executor(app, loop):
    app.ctx.executor = ThreadPoolExecutor(max_workers=4)

@app.after_server_stop
async def shutdown_executor(app, loop):
    app.ctx.executor.shutdown(wait=True)

# 确保目录存在
os.makedirs("upload", exist_ok=True)
os.makedirs("sentiment-analysis", exist_ok=True)


def analyser(id):
    try:
        in_path = os.path.join("upload")
        out_path = os.path.join("sentiment-analysis")
        comment_analyser = parse.Comment_analyser(in_path, out_path, id)
        comment_analyser.make_analyse()
    except Exception as e:
        print(f"影评分析失败：{str(e)}")


# 上传电影数据文件接口
@app.route("/v1/movie/crawled/upload", methods=['POST'])
async def upload(request):
    try:
        file = request.files.get('file')
        if not file:
            return res_json({"code": 0, "message": "文件未上传"}, ensure_ascii=False)

        # 验证文件类型
        filename = file.name
        _, ext = os.path.splitext(filename)
        if ext.lower() != '.json':
            return res_json({"code": 0, "message": "文件格式错误"}, ensure_ascii=False)

        # 生成唯一文件名
        now_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
        id = now_time + "_" + os.path.splitext(filename)[0]
        filename = id + ".json"
        path = os.path.join("upload", filename)

        # 保存文件
        with open(path, 'wb') as f:
            f.write(file.body)

        # 提交后台任务
        app.ctx.executor.submit(analyser, id)  # 直接提交不等待

        # 生成下载链接（id 即图片文件名前缀）
        download_link = []
        for ip in ips:
            link = f"http://{ip}:{port}/v1/image/download?filename={id}.zip"
            download_link.append(link)


        return res_json({
            "code": 1,
            "msg": "上传成功，分析任务已提交",
            "data": {
                "name": filename,
                "download_url": download_link
            }
        }, ensure_ascii=False)

    except Exception as e:
        return res_json({"code": 0, "msg": f"服务器错误: {str(e)}"}, ensure_ascii=False)


@app.route("/v1/image/download", methods=['GET'])
async def download_image(request):
    """下载生成压缩包图片"""
    try:
        filename = request.args.get("filename")
        if not filename:
            return res_json({"code": 0, "msg": "缺少 filename 参数"}, ensure_ascii=False)

        # 安全路径验证
        safe_path = os.path.abspath(os.path.join("sentiment-analysis", filename))
        if not safe_path.startswith(os.path.abspath("sentiment-analysis")):
            return res_json({"code": 0, "msg": "非法文件路径"}, ensure_ascii=False)

        # 检查文件是否存在
        if not os.path.exists(safe_path):
            return res_json({"code": 0, "msg": "文件不存在"}, ensure_ascii=False)

        # 返回文件并触发下载
        return await file(
            safe_path,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        return res_json({"code": 0, "msg": f"服务器错误: {str(e)}"}, ensure_ascii=False)


if __name__ == '__main__':
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        # 获取 IPv4 地址
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                ip = addr["addr"]
                if ip != "127.0.0.1":
                    ips.append(ip)
    app.run(host='0.0.0.0', port=port, workers=4, debug=False)