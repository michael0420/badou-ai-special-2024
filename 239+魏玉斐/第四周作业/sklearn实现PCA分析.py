from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
# 生成随机数据
data=np.random.random((20,4))
# 创建PCA对象，指定要降维到的维度
pca=PCA(n_components=2)

# 训练模型
pca.fit(data)

# 降维后的数据
reduced_data=pca.transform(data)
#画出原始数据的拟合图
plt.scatter(data[:,0],data[:,2],c='b',marker='o',label='Original data')

# 打印降维后的数据
print(reduced_data)

plt.show()