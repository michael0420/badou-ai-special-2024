import  random
import  cv2
import numpy as np

def f1(src,percetage):
	NoiseImg=src
	NoiseNum=int(percetage*src.shape[0]*src.shape[1])
	for i in range(NoiseNum):
		randX=random.randint(0,src.shape[0]-1)
		randY=random.randint(0,src.shape[1]-1)
		if random.random()<=0.5:
			NoiseImg[randX,randY]=0
		else:
			NoiseImg[randX,randY]=255

	return NoiseImg


img=cv2.imread("lenna.png",0)
img1=f1(img,0.5)
img2=cv2.imread("lenna.png",0)
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.waitKey()


