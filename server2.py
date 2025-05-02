from sanic import Sanic, response
from modelscope.pipelines import pipeline
import os, uuid, json

app = Sanic("SimpleSentiment")
app.config.TEMP_DIR = "./temp"

# 初始化模型（启动时需等待约30秒）
nlp = pipeline(
    'text-classification',
    'iic/nlp_structbert_sentiment-classification_chinese-base',
    device='cpu'
)

@app.post("/analyze")
async def analyze(request):
    # 文件验证
    if not (file := request.files.get('file')):
        return response.json({"error": "No file uploaded"}, status=400)
    
    # 保存临时文件
    file_id = str(uuid.uuid4())
    os.makedirs(app.config.TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(app.config.TEMP_DIR, f"{file_id}.jsonl")
    with open(temp_path, "wb") as f:
        f.write(file.body)
    
    try:
        results = []
        with open(temp_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                text = data.get("content", "").strip()
                if not text: continue
                
                # 执行分析
                res = nlp(text)
                results.append({
                    "text": text,
                    "positive": res['scores'][res['labels'].index('正面')],
                    "negative": res['scores'][res['labels'].index('负面')]
                })
        
        return response.json(results)
    
    except Exception as e:
        return response.json({"error": str(e)}, status=500)
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
