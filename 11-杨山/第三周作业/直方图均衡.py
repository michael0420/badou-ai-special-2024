#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
path="E:\BaiduNetdiskDownload\code\len.jpg"
img = cv2.imread(path, 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)


# In[ ]:





# In[ ]:




