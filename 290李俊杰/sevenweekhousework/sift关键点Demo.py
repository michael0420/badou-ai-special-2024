'''
3实现SIFT
'''


import cv2

img=cv2.imread("lenna.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 载入sift方法
sift=cv2.SIFT.create()
# 对图片使用SFIT方法提取关键点
# 检测关键点并计算特征描述子
kp,des=sift.detectAndCompute(img,None)
# 将结果绘制在原图上
'''
drawKeypoints(const Mat& image, const vector<KeyPoint>& keypoints,
 Mat& outImage, const Scalar& color = Scalar::all(-1), 
int flags = DrawMatchesFlags::DEFAULT )
参数说明：
image：输入图像，即要绘制关键点的图像。
keypoints：关键点向量，包含了检测到的关键点信息。
outImage：输出图像，在输入图像上绘制关键点后的结果图像。
color：关键点的颜色，默认为Scalar::all(-1)，表示随机颜色。
flags：绘制标志，用于设置绘制关键点的方式和样式，默认值为DrawMatchesFlags::DEFAULT。
DrawMatchesFlags::DEFAULT：默认方式，不进行特殊标记。
DrawMatchesFlags::DRAW_OVER_OUTIMG：在输出图像上绘制关键点，而不是在输入图像上绘制。
DrawMatchesFlags::DRAW_RICH_KEYPOINTS：绘制更丰富的关键点信息，例如显示关键点的方向和大小。
DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS：不绘制单个的关键点。
'''
img=cv2.drawKeypoints(image=img,outImage=img,keypoints=kp,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      color=(51, 163, 236))
# img=cv2.drawKeypoints(gray,kp,img)

cv2.imshow("img_sift",img)
cv2.waitKey(0)
