{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a2a5049",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 68 134 105]\n",
      "  [ 65 131 102]\n",
      "  [ 65 131 102]\n",
      "  ...\n",
      "  [108 113 158]\n",
      "  [110 115 160]\n",
      "  [110 115 160]]\n",
      "\n",
      " [[ 64 130 101]\n",
      "  [ 62 128  99]\n",
      "  [ 62 128  99]\n",
      "  ...\n",
      "  [101 107 152]\n",
      "  [102 108 153]\n",
      "  [102 108 153]]\n",
      "\n",
      " [[ 64 130 101]\n",
      "  [ 62 128  99]\n",
      "  [ 62 128  99]\n",
      "  ...\n",
      "  [101 107 152]\n",
      "  [102 108 153]\n",
      "  [102 108 153]]\n",
      "\n",
      " ...\n",
      "\n",
      " [[118 126 164]\n",
      "  [117 125 162]\n",
      "  [117 125 162]\n",
      "  ...\n",
      "  [139 147 184]\n",
      "  [150 158 195]\n",
      "  [150 158 195]]\n",
      "\n",
      " [[112 120 158]\n",
      "  [118 126 163]\n",
      "  [118 126 163]\n",
      "  ...\n",
      "  [141 149 186]\n",
      "  [137 145 182]\n",
      "  [137 145 182]]\n",
      "\n",
      " [[112 120 158]\n",
      "  [118 126 163]\n",
      "  [118 126 163]\n",
      "  ...\n",
      "  [141 149 186]\n",
      "  [137 145 182]\n",
      "  [137 145 182]]]\n",
      "(800, 800, 3)\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "def function(img):\n",
    "    height,width,channels =img.shape\n",
    "    emptyImage=np.zeros((800,800,channels),np.uint8)\n",
    "    sh=800/height\n",
    "    sw=800/width\n",
    "    for i in range(800):\n",
    "        for j in range(800):\n",
    "            x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整。\n",
    "            y=int(j/sw + 0.5)\n",
    "            emptyImage[i,j]=img[x,y]\n",
    "    return emptyImage\n",
    "    \n",
    "# cv2.resize(img, (800,800,c),near/bin)\n",
    "\n",
    "img=cv2.imread(\"qiche.png\")\n",
    "zoom=function(img)\n",
    "print(zoom)\n",
    "print(zoom.shape)\n",
    "cv2.imshow(\"nearest interp\",zoom)\n",
    "cv2.imshow(\"image\",img)\n",
    "cv2.waitKey(0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "033cef72",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
