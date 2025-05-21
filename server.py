from sanic import Sanic,response
from sanic.response import json,text,file
import os
import time
import json as ger_json
from pie_chart200 import generate_movie_pie_chart
from word_cloud import generate_movie_wordcloud
from emotion_analysis200 import analysis
from file_zip import file_zip

app=Sanic("mysanic")

@app.route("v1/book/crawled/upload",methods=['POST'])
async def upload(request):

    #错误处理代码
    allow_type=['.json']
    file=request.files.get('file')#?
    type=os.path.splitext(file.name)
    if len(type)==1 or type[1] not in allow_type:
        return json({"code": 0, "message":"file's format is error!"})

    #上传文件
    a_path="./upload"
    now_time=time.strftime("%Y%m%d%H%M%S",time.localtime())
    filename=now_time+'_'+type[0]+'.json'
    path=a_path+'/'+filename
    with open(path,"wb") as f:
        f.write(file.body)
    #情感分析和可视化显示
    analysis(path)
    generate_movie_pie_chart("emotion.json")
    generate_movie_wordcloud(path, "useless_word.txt")
    with open(path, "r", encoding='utf-8') as f:
        movie_comment_dic = ger_json.load(f)
    file_list = ["./image_file/"+name for name in list(movie_comment_dic.keys())]
    zip_path=now_time+"_image.zip"
    file_zip(zip_path, file_list)

    return json({'code':1,'msg':"upload successfully!",'data':{"comments_file": filename,"image.zip":zip_path}})

@app.route("/v1/movie/comment/sentiment-analysis/package",methods=['GET'])
async def download_images(request):
    now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    download_name=now_time+"_image.zip"
    file_path=f"./{request.args.get("zip_path")}"
    if not os.path.exists(file_path):
        return json({"code": 0, "message": "文件不存在!"}, status=404)
    return await file(
        file_path,
        filename=download_name
    )

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8000)
