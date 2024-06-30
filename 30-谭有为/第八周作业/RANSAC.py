import numpy as  np
import scipy as  sp
import scipy.linalg as sl
import pylab

#创建输入模型类
class LinearLeastSquareModel:
    #实例的初始化
    def __init__(self,input_colomns,output_colomns):
        self.input_colomns=input_colomns
        self.output_colomns=output_colomns

    #使用最小二乘法求最小平方和向量
    def fit(self,datas):
        A=np.vstack([datas[:,i] for i in self.input_colomns]).T #第一列数据
        B=np.vstack([datas[:,i] for  i in self.output_colomns]).T #第二列数据
        #scipy.linalg.lstsq函数，计算最小二乘解，# 返回值x，res 分别代表 回归系数，残差平方和，自变量X的秩、X的奇异值，
        # 参考文档：https://blog.csdn.net/weixin_43544164/article/details/122350501
        x,res,rank,s=sl.lstsq(A,B)
        return x

    #求残差平方和
    def get_err(self,datas,model):
        A=np.vstack([datas[:,i] for i in self.input_colomns]).T  #第一列转行
        B=np.vstack([datas[:,i] for  i in self.output_colomns]).T #第二列转行
        B_fit=sp.dot(A,model)  #将x代入最小二乘法获得的模型中得到y1
        err_per_point=np.sum((B-B_fit)**2,axis=1)   #（y-y1）的平方和
        return err_per_point

#随机在数据集中获取n个数据点idxs1 以及数据集中剩余点idxs2
def random_partition(n,n_datas):
        index=np.arange(n_datas)   #获取n_datas的索引
        np.random.shuffle(index)  #打乱索引
        idxs1=index[:n]
        idxs2=index[n:]
        return  idxs1,idxs2



#ransac函数，参数分别代表：输入数据集，参数模型，生成模型所需的最小样本点，最大迭代次数，阈值，拟合较好时需要的最小样本点个数
def ransac(datas,model,n,k,t,d):
     iterations=0  #迭代次数
     bestfit=None   #最优拟合解
     besterr=np.inf   #np.inf 表示正无穷大
     best_inlier_idxs=None
     while iterations<k:
         maybe_idxs,test_idxs=random_partition(n,datas.shape[0])  #在数据集中随机取出n个数据点maybe_idxs 和剩余的数据点test_idxs
         maybe_inliers=datas[maybe_idxs,:]   #获取n个数据点的具体数据（x，y）
         test_point=datas[test_idxs,:]  #获取其他数据点的具体数据
         maybe_model=model.fit(maybe_inliers)  #拟合模型
         test_err=model.get_err(test_point,maybe_model)   #使用拟合模型算出误差
         also_idxs=test_idxs[test_err<t]  #误差小于阈值算内群数据
         also_inliers=datas[also_idxs,:]
         if (len(also_inliers)>d):  #当内群数据大于最小样本点个数时
             betterdata=np.concatenate((maybe_inliers,also_inliers))  #总内群点数据个数
             bettermodel=model.fit(betterdata)  #算出新模型
             better_err=model.get_err(betterdata,bettermodel)   #用新模型算出新的误差
             thiserr=np.mean(better_err)  #当前模型的平均误差
             if thiserr<besterr:
                 bestfit=bettermodel
                 besterr=thiserr
                 best_inlier_idxs=np.concatenate((maybe_idxs,also_idxs))  #当前最大内群点，如后续循环中出现比这个内群点还大的，则替换
         iterations+=1
     if bestfit is None:
         raise ValueError("No suitable model found")
     else:
         return bestfit,best_inlier_idxs

#生成测试数据
data_num=500   #500个数据点
input_num=1  #输入变量个数
output_num=1   #输出变量个数

X_num=20*np.random.random(size=(data_num,1))  #生成500个在0-20之间的数字做完X坐标
#print(X_num)
rd_slope=20*np.random.normal(size=(1,1))   #随机生成一个斜率k
#print(rd_slope)
Y_num=sp.dot(X_num,rd_slope)    #y=kx
#print(Y_num)
X_noise=X_num+np.random.normal(size=X_num.shape)
Y_noise=Y_num+np.random.normal(size=Y_num.shape)   #x，y值加上高斯噪声
#print(X_noise.shape)
out_data_num=100   #100个局外点
index=np.arange(X_noise.shape[0])  #获取索引0-499
np.random.shuffle(index)  #打乱索引排序
out_index=index[:out_data_num]  #获取100个局外点的随机索引
X_noise[out_index]=20*np.random.random(size=(out_data_num,1))
Y_noise[out_index]=50*np.random.normal(size=(out_data_num,1))
#print(X_noise,Y_noise)
datas=np.hstack((X_noise,Y_noise))
input_colomns=range(input_num)   #数组的第一列=0
output_colomns=[input_num+i for i in range(output_num)]  #数组的最后一列=1
#print(input_colomns)
#print(output_colomns)

#将（500,1）的数组和（500,1）的数组堆叠成（500,500）的数组,将这个数组当做ransac的输入数据
print(datas)
model=LinearLeastSquareModel(input_colomns,output_colomns)  #类的实例化
x,res,rank,s=sl.lstsq(datas[:,input_colomns],datas[:,output_colomns])   #最小二乘法

#RANSAC，得到最佳模型和相应内群点
ransac_fit,ransac_data=ransac(datas,model,80,100,7e3,250)


#画图
X_num1=X_num[:,0]

pylab.plot(X_noise[:,0],Y_noise[:,0],'.',linestyle=' ',color='red',label='data')
pylab.plot(X_noise[ransac_data],Y_noise[ransac_data],'1',linestyle='',color='blue',label='ransac data')
pylab.plot(X_num1,np.dot(X_num,ransac_fit)[:,0],color='yellow',label='ransac_fit') #ransac出的模型
pylab.plot(X_num1,np.dot(X_num,rd_slope)[:,0],color='green',label='exact system')  #理想模型
pylab.plot(X_num1,np.dot(X_num,x)[:,0],color='black',label='linear fit')  #最小二乘法得到的模型
pylab.legend(loc='best')
pylab.show()



