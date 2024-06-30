import numpy as np
import cv2
from Utils import ImageUtils

class Sift(object):
    def __init__(self):
        pass

    def detect(self, img: np.ndarray, sigma=0.577):
        h, w = img.shape[0:2]
        num_octaves = int(np.log2(min(h, w)) - 3)
        print("num_octaves", num_octaves)
        GaussianPyramid, scale_sigma_map = self.generateGaussianPyramid(img, num_octaves=num_octaves,num_scales=6, sigma=sigma)
        for i in range(0, len(GaussianPyramid)):
            for j in range(0, GaussianPyramid[0].shape[0]):
                print(f"Gaussian: octave: {i}, scale: {j}, img_HeightxWidth: {GaussianPyramid[i].shape[1]}x{GaussianPyramid[i].shape[2]}")
        DoGPyramid = self.generateDoGPyramid(GaussianPyramid)
        for i in range(0, len(DoGPyramid)):
            for j in range(0, DoGPyramid[0].shape[0]):
                print(f"DoG: octave: {i}, scale: {j}, img_HeightxWidth: {DoGPyramid[i].shape[1]}x{DoGPyramid[i].shape[2]}")
        keyPointsList, keyPoints_dict = self.findKeyPoint(DoGPyramid)
        print(len(keyPointsList))
        # refinedKeyPointsList = self.refineKeyPoints(keyPointsList, DoGPyramid)
        #         # print(len(refinedKeyPointsList))
        # keypoints_with_orientations = self.assign_orientations(keyPointsList, GaussianPyramid)
        # print(len(keypoints_with_orientations))
        descriptors, keypoints_with_orientations = self.computeDescriptors(GaussianPyramid, keyPoints_dict, img)
        print(len(keypoints_with_orientations))
        print(descriptors.shape)
        keyPointsOutput = self.reconstructKeyPointsWithOrientations(keypoints_with_orientations, scale_sigma_map, img)
        print(len(keyPointsOutput))
        return keyPointsOutput, descriptors


    def generateGaussianPyramid(self, img: np.ndarray, num_octaves=4, num_scales=6, sigma=0.577):
        scale_sigma_map = {}
        for i in range(0, num_scales):
            if(i==0):
                scale_sigma_map[i] = 0
            else:
                k = (2 ** (1 / num_scales)) ** i
                sigma = sigma * k
                scale_sigma_map[i] = sigma
        h, w = img.shape[0:2]
        gaussianCollection = []   # shape is (group, layer, h, w)
        for i in range(0, num_octaves):
            if(i == 0):
                # layers0 = np.array(layerN, h*2, w*2)   # group_i = layers_i
                group_i = np.zeros((num_scales, h * 2, w * 2))  # group_i = layers_i
                group_i[0, :, :] = ImageUtils.nearestInteropate(img, (h * 2, w * 2))
            else:
                imgNew_shape = (int(gaussianCollection[-1].shape[1] / 2), int(gaussianCollection[-1].shape[2] / 2))
                group_i = np.zeros((num_scales, imgNew_shape[0], imgNew_shape[1]))  # group_i = layers_i
                group_i[0, :, :] = ImageUtils.nearestInteropate(gaussianCollection[-1][-3, :, :], imgNew_shape)
            for j in range(1, num_scales):
                gaussianFilterKernel = ImageUtils.generateGaussianFilter(sigma=scale_sigma_map[j], guassianFilterDim=3)
                group_i[j, :, :] = ImageUtils.convolutionFilterC1(group_i[j - 1, :, :], gaussianFilterKernel, stride=1, padding=1)
            gaussianCollection.append(group_i)
        return gaussianCollection, scale_sigma_map

    def generateDoGPyramid(self,gaussianPyramid):
        DoGCollection = []
        num_octaves =  len(gaussianPyramid)
        num_scales = gaussianPyramid[0].shape[0]  # layers(scales) count is same for each group(octave)
        for i in range(0, num_octaves):
            group_i = np.zeros((num_scales-1, gaussianPyramid[i].shape[1], gaussianPyramid[i].shape[2]))
            for j in range(0, num_scales-1):
                group_i[j, :, :] = gaussianPyramid[i][j+1, :, :] - gaussianPyramid[i][j, :, :]
            DoGCollection.append(group_i)
        return DoGCollection

    def findKeyPoint(self, DoGpyramid):
        keyPoints_list = []
        keyPoints_dict = {}
        num_octaves = len(DoGpyramid)
        num_scales = DoGpyramid[0].shape[0]  # layers(scales) count is same for each group(octave)
        for i in range(0, num_octaves):
            for j in range(1, num_scales-1):
                img_current = DoGpyramid[i][j,:,:]
                img_previous = DoGpyramid[i][j-1,:,:]
                img_next = DoGpyramid[i][j+1,:,:]
                keyPoints_dict[(i, j)] = []
                for m in range(1, img_current.shape[0]-1):
                    for n in range(1, img_current.shape[1]-1):
                        window_current = img_current[m-1:m+2, n-1:n+2]
                        window_previous = img_previous[m - 1:m + 2, n - 1:n + 2]
                        window_next = img_next[m - 1:m + 2, n - 1:n + 2]
                        window_current_maxIndex = np.argmax(window_current)
                        if(window_current_maxIndex == 4):
                            maxPixelValue_in_img_current = window_current[1, 1]
                            maxPixelValue_in_img_previous = np.max(window_previous)
                            maxPixelValue_in_img_next = np.max(window_next)
                            if(maxPixelValue_in_img_current>maxPixelValue_in_img_previous and maxPixelValue_in_img_current>maxPixelValue_in_img_next):
                                feature = [i, j, m, n]
                                keyPoints_list.append(feature)
                                keyPoints_dict[(i,j)].append((m,n))
                                continue
                        window_current_minIndex = np.argmin(window_current)
                        if (window_current_minIndex == 4):
                            minPixelValue_in_img_current = window_current[1, 1]
                            minPixelValue_in_img_previous = np.min(window_previous)
                            minPixelValue_in_img_next = np.min(window_next)
                            if(minPixelValue_in_img_current < minPixelValue_in_img_previous and minPixelValue_in_img_current < minPixelValue_in_img_next):
                                feature = [i, j, m, n]
                                keyPoints_list.append(feature)
                                keyPoints_dict[(i, j)].append((m, n))
        return keyPoints_list, keyPoints_dict

    def refineKeyPoints(self, keyPoints, DogPyramid):
        refined_keypoints = []
        for group_idx, layer_idx, i, j in keyPoints:
            offset, contrast = self.compute_subpixel_offset(DogPyramid[group_idx], layer_idx, i, j)
            if abs(contrast) > 0.6:
                refined_keypoints.append((group_idx, layer_idx, i + offset[0], j + offset[1]))
        return refined_keypoints

    def compute_subpixel_offset(self, group_i, layer_idx, i, j):
        di = (group_i[layer_idx, i+1, j] - group_i[layer_idx, i-1, j] ) * 0.5
        dj = (group_i[layer_idx, i, j+1] - group_i[layer_idx, i, j-1] ) * 0.5
        ds = (group_i[layer_idx+1, i, j] - group_i[layer_idx-1, i, j] ) * 0.5
        contrast = group_i[layer_idx,i,j] + 0.5 * (di * di + dj * dj + ds * ds)
        offset = np.array([di, dj, ds])
        return offset, contrast

    def assign_orientations(self, keypoints, gaussian_pyramid):
        keypoints_with_orientations = []
        for octave_idx, scale_indx, i, j in keypoints:
            gaussian_image = gaussian_pyramid[octave_idx][scale_indx,:,:]
            dx = gaussian_image[i+1, j] - gaussian_image[i-1, j]
            dy = gaussian_image[i, j+1] - gaussian_image[i, j-1]
            magitude = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            keypoints_with_orientations.append([octave_idx, scale_indx, i, j, theta])
        return keypoints_with_orientations

    def compute_gradients(self, image, y, x, window_size=4):
        half_size = window_size // 2
        region = image[int(y - half_size):int(y + half_size + 1), int(x - half_size):int(x + half_size + 1)]

        if region.shape[0] != window_size + 1 or region.shape[1] != window_size + 1:
            return np.array([]), np.array([])

        # filter by sobel, and get gradient, tangent
        sobelFilter = np.zeros((3, 3, 2))
        sobelFilter[:, :, 0] = ImageUtils.sobelX
        sobelFilter[:, :, 1] = ImageUtils.sobelY
        img_sobel = ImageUtils.convolutionFilterC1Multible(image, sobelFilter, stride=1, padding=1)
        img_gd = np.sqrt(img_sobel[:, :, 0] ** 2 + img_sobel[:, :, 1] ** 2)
        for i in range(0, img_sobel[:, :, 0].shape[0]):
            for j in range(0, img_sobel[:, :, 0].shape[1]):
                if (np.abs(img_sobel[i, j, 0]) <= 1e-13):
                    img_sobel[i, j, 0] = 1e-12
        tangent = img_sobel[:, :, 1] / img_sobel[:, :, 0]
        dx = img_sobel[:, :, 0]
        dy = img_sobel[:, :, 1]
        # 计算梯度的幅度和方向
        magnitudes = np.sqrt(dx ** 2 + dy ** 2)
        orientations = np.arctan2(dy, dx)

        return magnitudes, orientations

    def computeDescriptors(self, GaussianPyramid, keyPoints_dict, image):
        keypoints_with_orientations = []
        (h, w) = image.shape[0:2]
        descriptors = []
        bin_num = 8
        half_size = 8
        for k,v in keyPoints_dict.items():
            gaussian_image = GaussianPyramid[k[0]][k[1]]
            gx = ImageUtils.convolutionFilterC1(gaussian_image, ImageUtils.sobelX, stride=1, padding=half_size)
            gy = ImageUtils.convolutionFilterC1(gaussian_image, ImageUtils.sobelY, stride=1, padding=half_size)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            angle = np.arctan2(gy, gx) * (180 / np.pi) % 360  # 将角度转换为0-360度范围
            for i, j in keyPoints_dict[k]:
                dx = gaussian_image[i + 1, j] - gaussian_image[i - 1, j]
                dy = gaussian_image[i, j + 1] - gaussian_image[i, j - 1]
                magitude = np.sqrt(dx ** 2 + dy ** 2)
                theta = np.arctan2(dy, dx)
                keypoints_with_orientations.append([k[0], k[1], i, j, theta])

                # if i > h - 1 or j > w - 1:  # prevent index of original image from exceeding
                #     continue
                # 提取关键点周围的16x16窗口
                patch_mag = mag[i+half_size - half_size:i+half_size + half_size, j+half_size - half_size:j+half_size + half_size]
                patch_angle = angle[i+half_size - half_size:i+half_size + half_size, j+half_size - half_size:j+half_size + half_size]

                # 初始化描述符
                descriptor = np.zeros((4, 4, bin_num))
                # 分为4x4网格，每个网格生成方向直方图
                cell_size = half_size*2 // 4
                for i in range(4):
                    for j in range(4):
                        cell_mag = patch_mag[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                        cell_angle = patch_angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                        hist, hist_edge = np.histogram(cell_angle, bins=bin_num, range=(0, 360), weights=cell_mag)
                        # hist, hist_edge = np.histogram(cell_angle, bins=bin_num, range=(0, 360) )
                        descriptor[i, j, :] = hist
                descriptors.append(descriptor.flatten())

        return np.array(descriptors).astype(np.float32), keypoints_with_orientations  # openCV bf matcher need CVF32 data type

    def reconstructKeyPointsWithOrientations(self, keypoints_with_orientations, scale_sigma_map, image):
        (h, w) = image.shape[0:2]
        keyPointOutput = []
        for octave_idx, scale_indx, i, j, theta in keypoints_with_orientations:
            # if i > h-1 or j > w-1:  # prevent index of original image from exceeding
            #     continue
            if(octave_idx == 0):
                keyPointOutput.append([int(i/2), int(j/2), scale_sigma_map[scale_indx], theta])  # [i,j, sigma, theta]
            else:
                keyPointOutput.append([int(i*octave_idx), int(j*octave_idx), scale_sigma_map[scale_indx], theta])  # [i,j, sigma, theta]

        return keyPointOutput

    def drawKeyPoints(self, keypoints, img, imgName):
        for i,j,sigma,theta in keypoints:
            x = j
            y = i
            cv2.circle(img, center=(x, y), radius=int(sigma), color=(0, 255, 0), thickness=2)
        cv2.imshow(imgName + ' with keyPoints', img)

    def drawKeyPoints_CV(self, keypoints, img, imgName):
        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                color=(51, 163, 236))
        cv2.imshow(imgName+ ' keypoints', img)

    def detect_CV(self, greyImg):
        sift = cv2.xfeatures2d.SIFT_create()  # openCV existed lib
        keypoints, descriptor = sift.detectAndCompute(greyImg, None)
        return keypoints, descriptor

    def descriptorsMatch_CV(self, descriptor1, descriptor2):
        bf = cv2.BFMatcher(cv2.NORM_L2)  # openCV existed lib, also see flaNN
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)  # openCV bf matcher need CVF32 data type
        goodMatch = []
        for m, n in matches:
            # print(m.distance, n.distance)
            # print(m.queryIdx, n.queryIdx)
            # print(m.trainIdx, n.trainIdx)

            if m.distance < 0.50 * n.distance:
                goodMatch.append(m)
        return goodMatch

    def drawMatches_my_ij(self, img1, kp1, img2, kp2, goodMatch, imgName):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2

        p1 = [kpp.queryIdx for kpp in goodMatch]
        p2 = [kpp.trainIdx for kpp in goodMatch]

        post1 = np.int32([kp1[pp][0:2] for pp in p1])
        post2 = np.int32([kp2[pp][0:2] for pp in p2]) + (0, w1)   # mat row-col (i, j) order = opencv point (y, x) order

        for (i1, j1), (i2, j2) in zip(post1, post2):
            cv2.line(vis, (j1, i1), (j2, i2), (0, 0, 255))   # (x, y) order tuple -> (j, i)

        # cv2.namedWindow("match",cv2.WINDOW_NORMAL)
        cv2.imshow(imgName + " match", vis)

    def drawMatches_CV_xy(self, img1, kp1, img2, kp2, goodMatch, imgName):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2

        p1 = [kpp.queryIdx for kpp in goodMatch]
        p2 = [kpp.trainIdx for kpp in goodMatch]

        post1 = np.int32([kp1[pp].pt for pp in p1])
        post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)   # opencv point (x, y) order = mat row-col (i, j) order

        for (x1, y1), (x2, y2) in zip(post1, post2):
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))   # (x, y) order tuple

        # cv2.namedWindow("match",cv2.WINDOW_NORMAL)
        cv2.imshow(imgName + " match", vis)