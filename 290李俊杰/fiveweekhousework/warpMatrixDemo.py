'''
【第五周作业】
2.实现透视变换
'''
import cv2
import numpy as np

img=cv2.imread("photo1.jpg")
# 得到原图上的四个坐标点和对应到新图上的坐标点生成透视矩阵
src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
src = np.array(src)
dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
dst = np.array(dst)
# print(img.shape)
# 记录点的数量
num=src.shape[0]
# 根据公式建立A，B两个矩阵
A=np.zeros((num*2,8))
A1=np.zeros((num*2,8))
B=np.zeros((num*2,1))
B1=np.zeros((num*2,1))
# 根据图表在矩阵中的每个坐标上填入对应的值
A1[0]=[src[0][0],src[0][1],1,0,0,0,-src[0][0]*dst[0][0],-src[0][1]*dst[0][0]]
A1[1]=[0,0,0,src[0][0],src[0][1],1,-src[0][0]*dst[0][1],-src[0][1]*dst[0][1]]
A1[2]=[src[1][0],src[1][1],1,0,0,0,-src[1][0]*dst[1][0],-src[1][1]*dst[1][0]]
A1[3]=[0,0,0,src[1][0],src[1][1],1,-src[1][0]*dst[1][1],-src[1][1]*dst[1][1]]
A1[4]=[src[2][0],src[2][1],1,0,0,0,-src[2][0]*dst[2][0],-src[2][1]*dst[2][0]]
A1[5]=[0,0,0,src[2][0],src[2][1],1,-src[2][0]*dst[2][1],-src[2][1]*dst[2][1]]
A1[6]=[src[3][0],src[3][1],1,0,0,0,-src[3][0]*dst[3][0],-src[3][1]*dst[3][0]]
A1[7]=[0,0,0,src[3][0],src[3][1],1,-src[3][0]*dst[3][1],-src[3][1]*dst[3][1]]

# print(dst)
B1[0]=dst[0][0]
B1[1]=dst[0][1]
B1[2]=dst[1][0]
B1[3]=dst[1][1]
B1[4]=dst[2][0]
B1[5]=dst[2][1]
B1[6]=dst[3][0]
B1[7]=dst[3][1]

# print(B1)
for i in range(0, num):
    A_i = src[i, :]
    B_i = dst[i, :]
    A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                   -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
    B[2 * i] = B_i[0]

    A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                       -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
    B[2 * i + 1] = B_i[1]
# print(A==A1)
# print(B==B1)
#根据公式A*warpMatrix=B得出warpMatrix=B/A  =A的逆矩阵*B
# print(A1)
A1 = np.mat(A1)
# print(A1)
warpMatrix = A1.I * B1
# A1是8*8 B1是8*1 结果为8*1矩阵，
# 但是透视变换的通用公式中warpMatrix是3*3的矩阵所以要进行矩阵转换
# print(warpMatrix)
#之后为结果的后处理
#这里将warpMatrix转换成1*8的矩阵
warpMatrix = np.array(warpMatrix).T[0]
# print(warpMatrix)
#将之前为了简化公式而舍去的a33加入回来变成1*9的矩阵
warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)

# print(warpMatrix)
# 最后将warpMatrix转换成通用公式里面的3*3矩阵
warpMatrix = warpMatrix.reshape((3, 3))
print(warpMatrix)

