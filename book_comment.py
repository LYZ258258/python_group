from sanic import Sanic
from sanic.response import json
import os
import time

app = Sanic("mySanic")

# 确保上传目录存在
os.makedirs("uploads", exist_ok=True)

# 上传书评数据文件接口
@app.route("/v1/book/crawled/upload", methods=['POST'])
async def upload(request):
    try:
        allow_type = ['.json']  # 允许上传的类型
        file = request.files.get('file')  # 解析前端传来的文件
        type = os.path.splitext(file.name)  # 分割文件名
        if not file:
            return json({"code": 0, "message": "文件未上传"})
        if len(type) == 1 or type[1] not in allow_type:  # 查看是否为JSON文件
            return json({"code": 0, "message": "文件格式错误"})

        # 保存文件
        path = f"./uploads"
        now_time = time.strftime('%Y%m%d%H%M%S', time.localtime())  # 获取当前时间
        filename = now_time + "_" + type[0] + ".json"

        with open(path + "/" + filename, 'wb') as f:
            f.write(file.body)

        return json({"code": 1, "msg": "上传成功", "data": {"name": filename}})
    except Exception as e:
        return json({"code": 0, "msg": f"服务器错误: {str(e)}"}, status=500)

# 获取图书信息
@app.route("/v1/book/info", methods=['GET'])
async def get_books_info(request):
    book_id = request.args.get('book_id')
    if not book_id:
        return json({"code": 40002, "msg": "缺少 book_id 参数"}, status=400)

    # 示例数据，替换为实际数据库查询
    book_info = {"id": book_id, "title": "示例图书"}
    return json({"code": 200, "msg": "成功", "data": book_info})

# 获取书评信息
@app.route("/v1/book/comment", methods=['GET'])
async def get_book_comments(request):
    book_id = request.args.get('book_id')
    if not book_id:
        return json({"code": 40002, "msg": "缺少 book_id 参数"}, status=400)

    # 示例数据，替换为实际查询
    comments = [{"id": 1, "content": "好评！"}]
    return json({"code": 200, "msg": "成功", "data": comments})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, workers=4, debug=False)