from numpy import random
import numpy as  np

MySample=random.randint(0,100,(10,3)).astype(np.float) #随机生成10行三列的矩阵,值为浮点型,在0-100之间
np.set_printoptions(precision=2)  #矩阵值保留两位小数
print('随机生成的矩阵为：',MySample)
#print(type(MySample))
'''
MySample = np.array([[10.0, 15.0, 29.0],
                  [15.0, 46.0, 13.0],
                  [23.0, 21.0, 30.0],
                  [11.0, 9.0,  35.0],
                  [42.0, 45.0, 11.0],
                  [9.0,  48.0, 5.0],
                  [11.0, 21.0, 14.0],
                  [8.0,  5.0,  15.0],
                  [11.0, 12.0, 21.0],
                  [21.0, 20.0, 25.0]])
 '''
dim1=MySample[:,0]  #矩阵切片，取所有行的第一列
dim2=MySample[:,1]
dim3=MySample[:,2]
print(dim1,dim2,dim3)

avg=[]
Sdim1=np.mean(dim1)  #mean函数  算第一列的平均值
Sdim2=np.mean(dim2)
Sdim3=np.mean(dim3)
#print(type(Sdim1))
avg.append(Sdim1)
avg.append(Sdim2)
avg.append(Sdim3)
print(avg)   #算出各列平均值并存到avg列表中

for i in range(10):
    for j in range(3):
        MySample[i,j]=MySample[i,j]-avg[j]  #原矩阵的每一个数都减去每列的平均值，得到中心化后的矩阵
print('中心化后的矩阵为：',MySample)
print(np.shape(MySample)) #shape函数，得到矩阵的（行数，列数）

MySample_covariance_matrix=np.dot(MySample.T,MySample)/np.shape(MySample)[0]
#根据公式 D=（1/m）*Z^T*Z  意思是中心化后的矩阵Z* Z的转置矩阵除以样本个数即矩阵行数,注意一定是Z^T*Z,而不是Z*Z^T,m表示样本个数即矩阵的行数
print('协方差矩阵为：',MySample_covariance_matrix)

a,b=np.linalg.eig(MySample_covariance_matrix) #通过numpy的eig接口，得到矩阵的特征值a和特征向量b
print('特征值为：',a)
print('特征向量为',b)
c=np.argsort(-1*a)   #排序函数，-1表示倒序
print('倒序排列特征值后的特征值',c)
k=2  #设置降维后的维度为2
UT = [b[:,c[i]] for i in range(k)]   #取前两个特征值对应的特征向量
print('排序后前两列的特征向量为：',UT)
U=np.transpose(UT)   #矩阵转置，转换为2维矩阵
print('降维转换矩阵为：',U)    #降维成功。三维矩阵变为了二维矩阵，PCA实现




