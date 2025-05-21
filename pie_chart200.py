import json
import matplotlib.pyplot as plt
import os


def generate_movie_pie_chart(emotion_path):
    # 指定中文字体
    plt.rcParams["font.family"] = ["SimHei", "KaiTi", "STSong", "STKaiti"]

    with open(emotion_path, "r", encoding="utf-8") as f:
        mapdata_json = json.load(f)

    for movie_name, level_list in mapdata_json.items():
        os.makedirs("./image_file/"+movie_name, exist_ok=True)

        positive_level = level_list["positive_level"]
        negative_level = level_list["negative_level"]

        # 创建新的图形对象，设置合适的大小
        plt.figure(figsize=(6, 6))  # 调整图形大小

        # 饼图数据
        labels = ['Positive', 'Negative']
        sizes = [positive_level, negative_level]
        colors = ['skyblue', 'coral']

        # 绘制饼图
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(movie_name)

        # 保存图形
        safe_movie_name = movie_name.replace("/", "_")  # 避免文件名包含非法字符
        save_path = f"./image_file/{movie_name}/《{safe_movie_name}》情感分析图.png"
        plt.savefig(save_path)

        # 关闭当前图形，释放资源
        plt.close()
if __name__ == "__main__":
    generate_movie_pie_chart("emotion.json")


