# 数据集聚类
from sklearn.cluster import KMeans
import matplotlib.pyplot as pit

# 设置中文
pit.rcParams['font.family'] = 'SimHei'

# 原始数据
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
     ]


# kmeans聚类
def kmeansTest(k):
    # 创建KMeans对象
    kmeans_obj = KMeans(n_clusters=k)
    # 预测
    y_pred = kmeans_obj.fit_predict(X)
    # 打印聚类预测结果
    print("数据集聚类预测结果", y_pred)
    return y_pred


def draw_kmeans(result):
    # 获取数据集的第一列和第二列数据
    x = [n[0] for n in X]
    y = [n[1] for n in X]

    # 绘制散点图,x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
    pit.scatter(x, y, c=result, marker='x')
    # 绘制标题
    pit.title("kmeans聚类")
    # 绘制x轴和y轴坐标
    pit.xlabel("助攻数")
    pit.ylabel("得分数")

    # 设置右上角图例
    pit.legend(['A', 'B', 'C'])

    # 显示图像
    pit.show()


if __name__ == '__main__':
    draw_kmeans(kmeansTest(3))
