

import math
import random
import copy



def kmeans(data, k):
    """
    kmeans算法
    :param data: 数据
    :param k: 聚类数量
    :return: 聚类结果
    """

    # 随机选择k个点作为中心点
    center = random.sample(data, k)
    # print(center)
    # 创建一个空字典
    dic = {}
    for c in center:
        dic[c] = []

    flag = 0
    pre_dic = {}

    while True:
        print(center)

        # 遍历数据
        for i in data:
            dis_dic = {}
            # 遍历中心点
            for c in center:
                # 计算距离
                dis_dic[c] = abs(i - c)

            # 计算最小距离
            dis_min = min(dis_dic.values())

            # 将最小值加入中心点的数组
            for c in dis_dic.keys():
                if dis_dic[c] == dis_min:
                    dic[c].append(i)
                    break
        
        flag += 1
        print(f"第{flag}次迭代：")
        print(f"最新结果：{dic}")
        print(f"上次结果：{pre_dic}")
        
        if len(dic.keys()) > 0 and dic == pre_dic:
            break

        # 重新计算质心
        center=[]
        for c in dic.keys():
            center.append(sum(dic[c]) / len(dic[c]))

        # 存储上一次结果，深拷贝
        pre_dic = copy.deepcopy(dic)

        # 重置dic
        dic = {}
        for c in center:
            dic[c] = []

    print(f"=====累计迭代次数：{flag}=====")

    print(f"最终结果：{dic}")

    import matplotlib.pyplot as plt

    # 通过plt显示数据聚集情况
    for key in dic.keys():
        plt.plot(dic[key], label=key)
    plt.show()
            

data = [1.2, 3.5, 2.1, 5.1, 3.9, 5.5,9.1,12,3,10.1]
kmeans(data, 3)