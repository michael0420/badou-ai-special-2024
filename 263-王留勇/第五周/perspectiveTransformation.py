"""
寻找顶点 && 透视变换
"""

import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# 寻找顶点
def findAcme():
	gaussianImg = cv2.GaussianBlur(gray, (5, 5), 0)
	dilateImg = cv2.dilate(gaussianImg, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
	edged = cv2.Canny(dilateImg, 30, 120, 3)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0]
	docCnt = None

	if len(cnts) > 0:
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		for c in  cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02*peri, True)
			if len(approx) == 4:
				docCnt = approx
				break
	for peak in docCnt:
		peak = peak[0]
		cv2.circle(img, tuple(peak), 10, (255, 0, 0))

	cv2.imshow('img', img)
	cv2.waitKey(0)



# 透视变换
def transformation():
	src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
	dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
	print(img.shape)

	m = cv2.getPerspectiveTransform(src, dst)
	print(m)
	result = cv2.warpPerspective(img, m, (337, 488))
	cv2.imshow('result', result)
	cv2.waitKey(0)

if __name__ == '__main__':

	# 寻找顶点
	findAcme()

    # 透视变换
    # transformation()
