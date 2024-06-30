from sklearn.cluster import DBSCAN
import numpy as np
from PIL import Image as image
import cv2
def loadData(filePath):
    f = open(filePath,'rb')
    data= []
    img =image.open(f)
    m,n =img.size
    for i in range(m):
        for j in range(n):
            x,y,z =img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return np.mat(data),m,n
imgData,row,col = loadData('apple.jpg')

label = DBSCAN(eps=0.005,min_samples=1).fit_predict(imgData)
label=label.reshape([row,col])
pic_new = image.new("L",(row,col))
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
pic_new.save("DBSCAN_apple.jpg","JPEG")
img = cv2.imread("DBSCAN_apple.jpg")
cv2.imshow("DBSCAN_apple", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
