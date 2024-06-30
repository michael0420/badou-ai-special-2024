import cv2 as cv
from skimage import util
img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='speckle')  # mode 选定噪声类型
cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()