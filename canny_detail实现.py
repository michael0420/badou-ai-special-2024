import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    pic_path = "C:/Users/86188/Pictures/lenna.png"
    img = plt.imread(pic_path)
    print("image",img)
    if pic_path[-4:] == ".png":  #.png图片在这里的存储格式是0到1的浮点数，所以更扩展到255再计算
        img = img * 255   #还是浮点数类型
    img = img.mean(axis=-1)  #取均值的方法进行灰度化
    """
    这段代码实现了将一张彩色图片转换为灰度图片的过程。具体来说，它做了以下事情：

    导入了必需的库：numpy用于数组操作，matplotlib.pyplot用于图像操作。 定义了图片的文件路径pic_path。使用plt.imread()
    函数读取图片文件，并将其存储在变量img中。打印出读取的图片数据，用于查看或调试。检查图片文件是否为.png格式，
    如果是，则将图片的每个像素值乘以255，这是因为.png格式的像素值通常存储为0到1之间的浮点数，乘以255可以将其扩展到0到255的整数范围。
    使用mean(axis=-1)方法沿着最后一个轴（在本例中是颜色通道）取平均，从而将每个像素的RGB值转换为一个单一的灰度值。
    注意，乘以255的操作并不会改变数据类型，它仍然是浮点数类型。如果你需要得到整数类型的图像数据，你可以使用NumPy的astype()
    方法来转换数据类型，例如img = img * 255.astype(np.uint8)。另外，这个代码并没有显示转换后的图像，如果你想看到转换效果，你需要使用
    plt.imshow()函数来显示图像。
    """

    #1.高斯平滑
    #sigma = 1.52 #高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5 # 高斯平滑时的高斯核参数，标准差，可调
    dim = 5 #高斯核尺寸
    Gaussian_filter = np.zeros([dim,dim])   #存储高斯核,这是数组不是列表了
    tmp = [i-dim//2 for i in range(dim)]
    #这行代码 tmp = [i-dim//2 for i in range(dim)] 使用列表推导式生成了一个新的列表 tmp。
    #range(dim) 会生成一个从 0 到 dim 之间的整数序列（包括 0，不包括 dim）。
    # 然后，i-dim//2 会对这个序列中的每个元素 i 执行减去 dim 的整数除以 2 的操作。
    #整数除法 // 返回的是商的整数部分，所以 dim//2 就是 dim 的整数部分的一半。
    #因此，列表 tmp 中的每个元素都是相对于列表索引 i 的中心位置的偏移量。
    #例如，如果 dim 是 5，那么 tmp 就会是 [0, -1, -2, -1, 0]。这个列表会在接下来的代码中用作高斯滤波器的索引。
    #简而言之，这行代码的作用是生成一个关于中心对称的索引列表，用于后续计算高斯滤波器的权重
    n1 = 1/(2*math.pi*sigma**2) #计算高斯核
    n2 = -1/(2*sigma**2) #这两行代码是套公式， math.pi就是数学符号派3.14
    for i in range(dim):
        for j in range (dim):
            Gaussian_filter[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
            #n1 和 n2 是之前计算好的归一化常数和指数部分的系数。
            #math.exp(n2*(tmp[i]**2+tmp[j]**2)) 是高斯函数的指数部分，它依赖于索引 i 和 j。
            #这里使用了 math 模块的 exp() 函数来计算指数。
            #n1*math.exp(n2*(tmp[i]**2+tmp[j]**2)) 是最终的高斯滤波器系数，它会被分配给 Gaussian_filter[i, j]。
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx,dy =img.shape
    img_new = np.zeros(img.shape)  #存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim//2
    img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),"constant") #边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8),cmap="gray")  # 此时的img_new是255的浮点数整型，强制类型转换才可以，gray灰阶
    plt.axis("off")
    """
    这部分代码完成了将高斯滤波器应用到图像上的操作，以下是代码的详细解释：
    dx, dy = img.shape：获取原始图像的维度（宽度和高度）。
    img_new = np.zeros(img.shape)：创建一个新的数组 img_new 来存储平滑后的图像，其大小与原始图像 img 相同。
    由于使用了 np.zeros(), 初始所有值都设置为0，等待后面进行计算。
    tmp = dim//2：计算 dim 除以 2 的整数部分，用于后续的图像填充操作。
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')：使用 NumPy 的 np.pad() 函数对原始图像进行填充，
    以确保在进行滤波操作时不会超出数组边界。这里使用 'constant' 填充方式，填充的常数默认为 0。
    双层循环 for i in range(dx): for j in range(dy):：遍历原始图像的每个像素。
    img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)：将高斯滤波器应用到以当前像素为中心的邻域上，并将结果存入 img_new 的对应位置。这里使用了 NumPy 的广播机制和向量化操作来加速计算。
    plt.imshow(img_new.astype(np.uint8), cmap='gray')：使用 Matplotlib 库将处理后的图像显示出来，这里将 img_new 强制转换为 uint8 类型，并指定灰度色彩映射 (cmap='gray')。
    plt.axis('off')：隐藏图像的轴，以便获得一个更干净的显示效果。
    最终，这个过程将显示出经过高斯平滑处理后的图像。根据具体的应用需求，可以选择合适的标准差 sigma 和高斯核尺寸 dim 来达到预期的平滑效果。
    """
    #2.求梯度，以下两个是滤波用的sobel矩阵（检测图像中的水平，垂直和对角边缘）
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    """
    这两行代码定义了用于计算图像梯度的 Sobel 算子（卷积核）。Sobel 算子是用来检测图像中水平、垂直和对角线边缘的常用工具。
    sobel_kernel_x 是一个 3x3 的矩阵，用于检测水平边缘。它的左右两侧的行分别是 [[-1, 0, 1]], [-2, 0, 2]] 和 [-1, 0, 1]，
    这意味着当这个算子在图像上滑动时，它会在左右两边分别减去和加上相应的像素值，从而检测出亮度函数在水平方向上的变化。
    sobel_kernel_y 同样是 3x3 的矩阵，用于检测垂直边缘。它的上下两行分别是 [[1, 2, 1], [0, 0, 0]] 和 [-1, -2, -1]，
    意味着它会在上下两边分别加上和减去相应的像素值，从而检测出亮度函数在垂直方向上的变化。
    在实际应用中，将这些 Sobel 算子应用于图像，通过计算与这些算子卷积的结果，可以得到图像在各个方向上的梯度，
    从而提取出图像的边缘特征。这些特征对于图像分割、物体识别和其他计算机视觉任务非常重要
    """
    img_tidu_x = np.zeros(img_new.shape)  #存储梯度图像
    img_tidu_y = np.zeros([dx,dy])  #这两个梯度图像是一样的，只是为了看的清楚方便理解，工作中写一个就可以
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), "constant")
    #这行代码 img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant') 使用 NumPy 的 np.pad() 函数对 img_new 图像进行了填充，
    #添加了一层像素边框。np.pad() 函数有三个参数：img_new：需要进行填充的原始图像。
    #((1, 1), (1, 1))：分别表示对图像的上下和左右方向进行填充，这里都填充了1个像素，上下和左右总共各填充了2个像素。
    #'constant'：指定填充方式为常数填充，即填充的像素值都是常数。在图像处理中，这种填充通常用于确保在进行滤波操作时不会超出图像边界，
    #避免了由于边缘像素值不足而导致的计算错误。
    #在这个例子中，使用常数值进行填充是因为 Sobel 算子需要访问相邻的像素来进行梯度计算，而填充可以提供这些相邻像素。
    #在对图像进行填充之后，就可以安全地应用 Sobel 算子来计算图像的梯度，而不需要担心会因为边缘像素的问题而在计算中出错。

    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)

    #这部分代码计算了图像在x方向和y方向的梯度，以及梯度的模。
    #img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)：
    #这一行计算了在x方向上的梯度。这里使用了 NumPy 的广播机制和向量化操作来计算每次滑动窗口对应的梯度值。
    #img_pad[i:i + 3, j:j + 3] 表示从 img_pad 中取出一个 3x3 的窗口，
    #然后这个窗口与 sobel_kernel_x 进行逐元素相乘，最后将乘积求和，得到该窗口在x方向上的梯度。
    #img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y): 类似地，这一行计算了在y方向上的梯度。
    #img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2): 这一行计算了梯度的模，
    #它表示了亮度函数在该点的变化率的大小。
    #通过这种方式，代码分别在x方向和y方向上应用了 Sobel 算子，并计算了梯度的模，得到了一个反映图像亮度函数变化情况的图像。
    #这个梯度模图像可以用于进一步的处理，比如边缘检测、图像分割或其他任何需要图像梯度信息的任务。

    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    """
    这部分代码实现了两个主要的操作：计算梯度的角度，并显示梯度模的图像。
    img_tidu_x[img_tidu_x == 0] = 0.00000001：这行代码将 img_tidu_x 中的所有零值替换为一个极小的非零值。
    这样做是为了避免在后续计算角度时出现除以零的情况。在计算梯度角度时，我们需要除以梯度的x分量，如果x分量为零，将会造成除数为零的错误。
    为了避免这个问题，我们用一个足够小的常数来代替零值。
    angle = img_tidu_y / img_tidu_x：这行代码计算了梯度的角度。由于 img_tidu_y 和 img_tidu_x 分别代表了梯度的y分量和x分量，
    因此它们的比值可以表示梯度的方向。注意，这里可能会有符号问题，因为如果x分量和y分量的符号不同，
    那么角度可能会有不同的象限，这需要在实际应用中考虑。
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')：这行代码显示了梯度模的图像。
    这里将 img_tidu 强制转换为 uint8 类型，并使用了灰度色彩映射 (cmap='gray')。
    plt.axis('off')：这行代码隐藏了图像的轴，使得图像显示更加整洁。
    在实际应用中，这些操作可能会有不同的目的。例如，梯度的角度可以用于边缘方向的判断，
    而梯度模的图像可以用于显示图像中亮度变化的强烈程度。这些信息对于理解和分析图像特征非常有用。
    """
    #3.非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否，这行代码 temp = img_tidu[i-1:i+2, j-1:j+2] 是在
    # 获取指定像素点 (i, j) 的邻域像素值。在这个例子中，它获取的是一个 3x3 的邻域，包含了中心像素和它的八个邻居。
    #这里的 img_tidu 是之前计算得到的梯度模图像。通过索引 i-1:i+2 和 j-1:j+2，代码截取了以(i, j)为中心的邻近 3x3 区域的像素值，
    #并将这些值存储在变量 temp 中。
    #在非极大值抑制的上下文中，这个邻域被用来检查中心像素是否在其邻域内具有局部最大值。如果中心像素的梯度值不是局部最大值，
    #则该像素将被抑制，即它的值将被设置为零。通过这种方式，非极大值抑制能够消除那些不是边缘的强噪声点，从而使真正的边缘更加清晰。

               num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
               num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
               if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                   flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    """
    这部分代码实现了非极大值抑制（Non-maximum Suppression）的算法。非极大值抑制是一种常用于边缘检测中的技术，
    它的目的是抑制那些不是局部最大值的梯度，从而使得真正的边缘点更加突出。
    以下是代码的详细解释：
    初始化一个和梯度模图像 img_tidu 形状相同的零数组 img_yizhi 来存储最终抑制后的结果。
    双层循环遍历梯度模图像 img_tidu 的内部像素（不包含边界像素），因为边界像素的邻域可能会受到填充的影响，可能导致错误的抑制。
    对于每个像素，根据梯度的角度 angle 分别进行不同的处理：
    如果梯度角度小于 -1 或者大于 1，则通过插值计算出斜坡函数的两个端点的值，并检查当前像素的梯度值是否大于这两个端点值。
    如果不是，则抑制该像素。
    如果梯度角度在 (0, 1) 或 (-1, 0) 范围内，则通过插值计算出对应直线的两个端点的值，
    并检查当前像素的梯度值是否大于这两个端点值。如果不是，则抑制该像素。
    如果经过上述条件判断后，当前像素未被抑制（即梯度值仍为局部最大），则将该像素值保存在 img_yizhi 中。
    使用 Matplotlib 库显示抑制后的图像，并设置为灰度显示。
    通过非极大值抑制的处理，可以使得原先可能存在的模糊区域或者不连续的边缘点变得清晰和连续，
    从而增强图像的边缘特征。在实际应用中，这个步骤通常会与其他边缘检测算法结合使用，以提升边缘检测的效果。
    """
    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    """
    这行代码 lower_boundary = img_tidu.mean() * 0.5 计算了梯度模图像 img_tidu 的平均值，
    并且将这个平均值乘以 0.5 来得到一个较低的阈值 lower_boundary。
    在边缘检测的过程中，这个较低的阈值将被用来检测那些相对较弱的边缘。这是通过比较每个像素的梯度值与这个阈值来实现的：
    如果一个像素的梯度值高于这个阈值，那么它就会被认为是边缘的一部分；如果低于这个阈值，那么它就不会被视为边缘。
    这种方法可以帮助保留图像中的细细节和弱边缘，同时抑制噪声。在实际应用中，这个阈值可以根据需要调整，以达到最佳的边缘检测效果。
    """
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
             img_yizhi[i, j] = 255
             zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

        # 绘图
        plt.figure(4)
        plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
        plt.axis('off')  # 关闭坐标刻度值
        plt.show()


