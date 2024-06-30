from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 实例化PCA对象，选择要降到的维数
pca = PCA(n_components=2)  # 降到2个主成分

# 对数据进行PCA降维
df_reduced = pca.fit_transform(df)

# 输出结果
print(df_reduced)