from sanic import Sanic
from sanic.response import json
import os

app = Sanic("mySanic")

# 确保上传目录存在
os.makedirs("uploads", exist_ok=True)

# 上传书评数据文件接口
@app.route("/v1/book/crawled/upload", methods=['POST'])
async def upload(request):
    try:
        uploaded_file = request.files.get('file')
        if not uploaded_file:
            return json({"code": 40001, "msg": "文件未上传"}, status=400)

        filename = uploaded_file.name
        file_path = f"uploads/{filename}"

        # 保存文件
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.body)

        # 处理文件（假设 convert_book 已实现）
        # convert_book(file_path)

        return json({"code": 200, "msg": "上传成功", "data": None})
    except Exception as e:
        return json({"code": 500, "msg": f"服务器错误: {str(e)}"}, status=500)

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