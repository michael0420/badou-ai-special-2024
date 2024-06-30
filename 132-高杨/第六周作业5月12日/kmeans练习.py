import numpy as np
import cv2
import  matplotlib.pyplot as plt






img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

row, col=img_gray.shape[:]
data = img_gray.reshape((row*col),1)
print(data.shape)
data =np.float32(data)

#停止条件
stopcondiation = (cv2.TermCriteria_EPS +
                  cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#设置标签
flags= cv2.KMEANS_RANDOM_CENTERS

#调用接口，
compactness,labels,centers = cv2.kmeans(data,4,None,stopcondiation,10,flags)

#生成最后图像
dst = labels.reshape((img.shape[0],img.shape[1]))


#显示图像
titles =['orgin','kmeansResult']
images =[img_gray,dst]

for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()







