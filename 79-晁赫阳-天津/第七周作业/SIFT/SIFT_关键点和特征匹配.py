import cv2
import numpy as np

def SIFT(image_path):#关键点绘画
    # 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建 SIFT 对象
    sift = cv2.xfeatures2d.SIFT_create()

    # 检测关键点并计算描述符
    keypoints, descriptor = sift.detectAndCompute(gray, None)

    # 绘制关键点
    img_with_keypoints = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                           color=(51, 163, 236))

    # 显示图像
    cv2.imshow('SIFT Keypoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 该函数用于绘制匹配结果
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    # 获取两张图像的高度和宽度
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    # 创建一张黑色图像，宽度为两张图像宽度之和，高度为两张图像高度的最大值
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    # 将两张图像分别放在新创建图像的左侧和右侧
    vis[:h1, :w1] = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
    vis[:h2, w1:w1 + w2] = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)

    # 获取匹配点的索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 获取匹配点的坐标
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    # 在两张图像之间绘制匹配线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return vis

# 该函数用于进行特征匹配
def feature_matching(img1_path, img2_path):
    # 读取两张灰度图像
    img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 使用SIFT算法检测和计算特征点及描述符
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # 使用BFMatcher进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 筛选出好的匹配点
    goodMatch = [m for m, n in matches if m.distance < 0.50 * n.distance]

    # 绘制匹配结果
    result_img = drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])
    return result_img


if __name__ == "__main__":
    image_path = "E:\\pycharm_code\\cv\\Original_Data\\0216.jpg"
    SIFT(image_path)

    # 设置输入图像的文件路径
    img1_path = "iphone1.png"
    img2_path = "iphone2.png"

    # 进行特征匹配并获取结果图像
    result_img = feature_matching(img1_path, img2_path)

    # 显示结果图像
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
