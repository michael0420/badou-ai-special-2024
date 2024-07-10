import cv2 as cv
from skimage import util
img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='speckle')  # mode 选定噪声类型
cv.imwrite("../lenna_noise.jpg",noise_gs_img*255,[int(cv.IMWRITE_JPEG_QUALITY),100])
cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()