from __init__ import say_hello, cv_imread, cv_set_titile, current_directory
import os
import cv2

def test():
    say_hello("田海龙")


def cv_img_test():
    img_path = os.path.join(current_directory, "img", "lenna.png")
    img = cv_imread(img_path)
    cv2.imshow("lenna", img)
    cv_set_titile("lenna", "原图")

    cv2.waitKey(0)


# cv_img_test()
    

def sactter_test():
    import matplotlib.pyplot as plt

    # 准备数据
    x = [1, 2, 3, 4, 5,1]
    y = [3.5, 4, 3, 2, 1,7]

    # 创建散点图
    plt.scatter(x, y,c=[0,0,1,2,2,1],marker="s")

    #用来正常显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei']
    # 添加标题和轴标签
    plt.title('散点图示例')
    plt.xlabel('x轴')
    plt.ylabel('y轴')

    # 显示图表
    plt.show()

# sactter_test()
    
import numpy as np

def array_test():
    c=[35,74,55,91]
    ss=[1,1,1,4,3,2,3,1,4,3,1,2,3]
    c=np.array(c)
    sss=[]
    for s in ss:
        s=s-1
        sss.append([s])
    

    print(sss)
    sss=np.array(sss)
    print(sss)
    print(sss.flatten())
    print(c[sss.flatten()])


array_test()
