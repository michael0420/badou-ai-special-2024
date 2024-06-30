import numpy as np
import cv2
import matplotlib.pyplot as plt



img = cv2.imread('lenna.png')
# 注意如果要用plt读取的话 要把每一个像素 *255
print(img)
row,col = img.shape[0],img.shape[1]
data = img.reshape((row*col),3)
print(data.shape)
#
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


flags = cv2.KMEANS_RANDOM_CENTERS

compactness,labels2,centers2=cv2.kmeans(data,2,None,criteria,10,flags)

compactness,labels4,centers4=cv2.kmeans(data,4,None,criteria,10,flags)

compactness,labels8,centers8=cv2.kmeans(data,8,None,criteria,10,flags)

compactness,labels16,centers16=cv2.kmeans(data,16,None,criteria,10,flags)

compactness,labels32,centers32=cv2.kmeans(data,32,None,criteria,10,flags)

compactness,labels64,centers64=cv2.kmeans(data,64,None,criteria,10,flags)

compactness,labels128,centers128=cv2.kmeans(data,128,None,criteria,10,flags)

centers2 = np.uint8(centers2)
print(centers2)
print(labels2)
print(labels2.flatten())
res = centers2[labels2.flatten()]
print(res)
dst2 = res.reshape((img.shape))


centers4 = np.uint8(centers4)
res4 = centers4[labels4.flatten()]
dst4 = res4.reshape((img.shape))


centers8 = np.uint8(centers8)
res8 = centers8[labels8.flatten()]
dst8 = res8.reshape((img.shape))


centers16 = np.uint8(centers16)
res16 = centers16[labels16.flatten()]
dst16 = res16.reshape((img.shape))


centers32 = np.uint8(centers32)
res32 = centers32[labels32.flatten()]
dst32 = res32.reshape((img.shape))


centers64 = np.uint8(centers64)
res64 = centers64[labels64.flatten()]
dst64 = res64.reshape((img.shape))


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst32 = cv2.cvtColor(dst32,cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)




images=[img,dst2,dst4,dst8,dst16,dst32,dst64]
titles=['orgin','kmeans2','kmeans4','kmeans8','kmeans16','kmeans32','kmeans64']
for i  in  range(7):
    plt.subplot(2,4,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()




