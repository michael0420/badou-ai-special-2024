import  numpy as  np
import matplotlib.pyplot as  plt


#归一化  -1~1  x_=(x−x_mean)/(x_max−x_min)
#归一化   0~1  x_=(x−x_min)/(x_max−x_min)
def normalization(datas):
     avg=np.mean(datas)
     res=[]
     for i in datas:
         res1=((float(i)-avg)/(max(datas)-min(datas)))
         res.append(res1)
     return res

#标准化    x_new=(x-u)/sigma    （原数值-平均值）/标准差
def z_score(datas):
    avg=np.mean(datas)
    for i in datas:
        sum=0
        sum=((float(i)-avg))**2+sum
    sigma=sum/(len(datas))
    res=[]
    for i in datas:
        res1=(float(i)-avg)/sigma
        res.append(res1)
    return res

datas=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

if __name__=='__main__':
    res=normalization(datas)
    res_z_score=z_score(datas)
    print(res)
    print(res_z_score)
    s=[]
    for i in datas:
        c=datas.count(i)
        s.append(c)
    print(s)
    plt.plot(res,s)
    plt.plot(res_z_score,s)
    plt.show()
