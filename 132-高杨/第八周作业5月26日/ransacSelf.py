import numpy as np
import scipy.linalg as sl

def ransac(model,data,n,k,d,t,return_all):
    # 第一步初始化所有的要的值
    bestfit = None  # 拟合的最好的一条直线
    besterror = np.inf # 最优的误差
    iterations =0   #迭代次数
    best_inside_num =0     #### 最重要的一个东西， 找出内群点最多的那个一个模型
    while iterations<k:
        # 划分内群点和其他点 , 注意要随机取
        n_inside,n_remain = InsidePointSearch(data,n)  # 返回的是下标
        n_inside_data = data[n_inside,:]      # 内群点的坐标 【x,y】
        n_remain_data = data[n_remain]
        # 拟合模型
        n_inside_model = model.fit(n_inside_data)
        n_inside_error = model.get_error(n_inside_model,n_remain_data)  #得到该模型的误差 ，执行步骤，将其余点加入运算，看其他点误差多少  和阈值t进行比较
        n_inside_out_idx = n_remain[n_inside_error<t]  # 获取剩余点的下标，这个下标是
        n_inside_outdata = data[n_inside_out_idx,:]
        if (len(n_inside_outdata)>d):
            betterdata = np.concatenate((n_inside_data,n_inside_outdata))
            beteermodel= model.fit(betterdata)
            bettererror = model.get_error(beteermodel,betterdata)
            currenterror  = np.mean(bettererror)
            if currenterror<besterror:
                besterror = currenterror
                bestfit = beteermodel
                best_inside_num = np.concatenate((n_inside,n_inside_out_idx))
        iterations+=1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'liner':best_inside_num}  #返回模型 ， 和一个字典，字典内容是拟合最好的内群点下标
    else:
        return bestfit

def InsidePointSearch(data,n):
    all_idx = np.arange(data.shape[0])   # [01,1,2,3,4,5,...]
    np.random.shuffle(all_idx)
    part_idx = all_idx[:n]
    remain_idx = all_idx[n:]
    return part_idx,remain_idx
class LinearLeastSquarlloos:
    def __init__(self,inp,outp):
        self.inp = inp
        self.outp = outp
    def fit(self,data):
        A = np.vstack([data[:,self.inp]] ).T   # 转置前 shape [ [1],[2]. [3]   ]  1*50
        B = np.vstack([data[:,self.outp]]).T
 #返回依次是 ，斜率和截距(一个对象，里面有两个值)、残差、秩和奇异值等输出
        scale,resides,rank,s = sl.lstsq(A,B)
        return scale
    def get_error(self,model,data):
        A = np.vstack([data[:, self.inp]]).T   # 转置前 vstack shape is 1*50
        B_fit = np.dot(A,model)       # model shape 1*1
        B = np.vstack([data[:, self.outp]]).T
        error = np.sum((B-B_fit)**2 , axis=1)
        return error

def test():
    #第一步，先生成数据
    n_data = 500
    n_input = 1
    n_output =1
    A_exact = 20* np.random.random((n_data,n_input))
    scale_exact = 60 * np.random.random(size=(n_input,n_output))
    B_exact = np.dot(A_exact,scale_exact)
    #加入高斯噪音点，方便处理
    A_Noise = A_exact + np.random.normal(size=(n_data,n_input))
    B_Noise = B_exact + np.random.normal(size=(n_data,n_input))
    #加入离群点，
    n_outsides = 100
    data_idx = np.arange(n_data)
    np.random.shuffle(data_idx)
    outsides_idx = data_idx[:n_outsides]
    A_Noise[outsides_idx] = 20 * np.random.random(size=(n_outsides,n_input))
    B_Noise[outsides_idx] = 50 * np.random.normal(size=(n_outsides,n_input))
    # 拼接在一起 做后续处理  【x,y】
    all_data = np.hstack((A_Noise,B_Noise))  #500*2

    temp_in=0  #送入 最小二乘 取0列
    temp_out=1
    model = LinearLeastSquarlloos(temp_in,temp_out)

    # 对数据只用最小二乘法拟合
    #用[0] 是由于必须传入二维的
    scl,resides,rank,s = sl.lstsq(all_data[:,[0]],all_data[:,[1]])
    return_all =True
    # ransac 方法拟合
    ransac_model,ransac_idx = ransac(model,all_data,50,1000,300,7e4,return_all)

    import pylab
    sorted_idx = np.argsort(A_exact[:,0])  # 对x从小到大排序并且返回下标
    sorted_data_x = A_exact[sorted_idx]   # 准确的A的坐标
    pylab.plot(A_Noise[:,0],B_Noise[:,0],'k.',label='data')  # 原始数据的data
    pylab.plot(A_Noise[ransac_idx['liner'],0],B_Noise[ransac_idx['liner'],0],'bx',label='Ransacdata')

    # 绘制拟合曲线
    pylab.plot(sorted_data_x[:,0],np.dot(sorted_data_x,ransac_model)[:,0],label='Ransac Model')
    pylab.plot(sorted_data_x[:,0],np.dot(sorted_data_x,scl)[:,0],label='LinerLeastSqural Model')
    pylab.plot(sorted_data_x[:,0],np.dot(sorted_data_x,scale_exact)[:,0],label='Exact Model')

    pylab.legend()
    pylab.show()

if __name__ == '__main__':
    test()



