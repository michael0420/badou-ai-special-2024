'''

【第6周作业】
 实现kemans（全)
# 第一种实现
'''
from sklearn.cluster import KMeans
"""
KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""
# 第一种实现，现实意义聚类，学生成绩表聚类
# 每一行代表一个学生，第一列代表期末成绩，
# 第二列代表数学成绩，每个科目总分都是一百
student_sorce1=[
    ["10","20"],
    ["20","30"],
    ["30","40"],
    ["40","50"],
    ["50","60"],
    ["60","70"],
    ["70","80"],
    ["70","100"],
    ["50","80"],
    ["60","20"],
    ["80","80"],
    ["70","40"],
    ["60","50"],
    ["50","100"],
    ["80","60"],
    ["90","40"],
    ["50","50"],
    ["50","70"],
    ["90","80"],
    ["70","70"],
]
student_sorce2=[
    ["10","0.05"],
    ["20","0.1"],
    ["30","0.15"],
    ["40","0.2"],
    ["50","0.25"],
    ["60","0.3"],
    ["70","0.35"],
    ["70","0.35"],
    ["50","0.25"],
    ["60","0.3"],
    ["80","0.4"],
    ["70","0.35"],
    ["60","0.3"],
    ["50","0.25"],
    ["80","0.4"],
    ["90","0.45"],
    ["50","0.25"],
    ["50","0.25"],
    ["96","0.45"],
    ["97","0.5"],
]
"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""
# 设定KMeans类簇数为2
sorce_fl=KMeans(n_clusters=2)
student_pred=sorce_fl.fit_predict(student_sorce1)
# print(student_pred)

# 使用可视化绘图更直观观测聚类结果
import matplotlib.pyplot as plt
#1.获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x=[]
y=[]
# print(len(student_sorce))
for i in range(len(student_sorce1)):
    x.append(student_sorce1[i][0])
    y.append(student_sorce1[i][1])

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=student_pred, marker='*')

# 绘制标题
plt.title("Kmeans-Student Sorce")

# 绘制x轴和y轴坐标
plt.xlabel("Chinese_Score")
plt.ylabel("Math_Score")

# 设置右上角图例
# plt.legend(["A", "B"])

# 显示图形
plt.show()