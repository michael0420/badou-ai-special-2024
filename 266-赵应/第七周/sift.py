import cv2

if __name__ == '__main__':
    img = cv2.imread("./image/test.jpg")
    # 创建特征点检测器
    sift = cv2.SIFT_create()
    # 提取特征点
    key_points, descriptors = sift.detectAndCompute(img, None)
    out_img = cv2.drawKeypoints(img, key_points, None)
    cv2.imshow("SIFT KeyPoints", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
