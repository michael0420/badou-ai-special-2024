import numpy as np
import cv2
import matplotlib.pyplot as plt
# 读图
img=cv2.imread('lenna.png')
# print(img.shape)

#获取图像的宽高
high=img.shape[0]
width=img.shape[1]
channel=img.shape[2]
# print(channel)

# 将像素集转换成一维
data=img.reshape((-1,3))
data=np.float32(data)
# print(data)

# 停止条件
criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

# 标志位
flags=cv2.KMEANS_RANDOM_CENTERS

# 设置聚类的数目
compactness,labels2,centers2=cv2.kmeans(data,2,None,criteria,10,flags)
compactness,labels4,centers4=cv2.kmeans(data,4,None,criteria,10,flags)
compactness,labels8,centers8=cv2.kmeans(data,8,None,criteria,10,flags)
compactness,labels16,centers16=cv2.kmeans(data,16,None,criteria,10,flags)
compactness,labels32,centers32=cv2.kmeans(data,32,None,criteria,10,flags)
compactness,labels64,centers64=cv2.kmeans(data,64,None,criteria,10,flags)

# 图像生成
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]     # 每一个像素点所代表的族的中心值
# dst=res.reshape((high,width,channel))
dst=res.reshape((img.shape))
# print(centers2)
# print(res.shape)
# print(res)
# print(dst)
centers4=np.uint8(centers4)
res4=centers4[labels4.flatten()]
dst4=res4.reshape((img.shape))

centers8=np.uint8(centers8)
res8=centers8[labels8.flatten()]
dst8=res8.reshape((img.shape))

centers16=np.uint8(centers16)
res16=centers16[labels16.flatten()]
dst16=res16.reshape((img.shape))

centers32=np.uint8(centers32)
res32=centers32[labels32.flatten()]
dst32=res32.reshape((img.shape))

centers64=np.uint8(centers64)
res64=centers64[labels64.flatten()]
dst64=res64.reshape((img.shape))

# 转RGB
dst=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
dst4=cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
dst8=cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
dst16=cv2.cvtColor(dst16,cv2.COLOR_RGB2BGR)
dst32=cv2.cvtColor(dst32,cv2.COLOR_RGB2BGR)
dst64=cv2.cvtColor(dst64,cv2.COLOR_RGB2BGR)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles=[u'原始图像', r'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 k=32',u'聚类图像 K=64']
images=[img,dst,dst4,dst8,dst16,dst32,dst64]
for i in range(7):
    plt.subplot(2,4, i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks(())
    plt.yticks(())

# plt.figure()
# plt.imshow(dst)
# plt.xticks(())
# plt.yticks(())
# plt.title('聚类图像')
plt.show()
