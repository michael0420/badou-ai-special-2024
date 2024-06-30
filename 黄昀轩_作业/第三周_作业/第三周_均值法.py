"""
@author: huangyunxuan
均质化直方图

"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#hist(x, bins=None, range=None, density=False,weights=None, cumulative=False,
#bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None,
#log=False, color=None, label=None, stacked=False, *, data=None, **kwargs)
# hist = cv2.calcHist([img_g],[0],None,[256],[0,256])

plt.hist(img_g.ravel(),256,[0,256])
plt.show()


