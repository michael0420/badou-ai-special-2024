第四周作业<br>
噪声 noise.py<br> 
pca pca.py<br>
<br>
设有N个n维数据向量组成的数据矩阵X，其中每一列代表一个原始数据向量。<br>
中心化后的数据矩阵表示为Z=X-E(X)，其中E(X)是X的均值向量。<br>
协方差矩阵定义为C=E{ZZ^T}，其中E{}表示期望，^T表示转置。<br>
将Z=X-E(X)带入得：<br>
C=E{(X-E(X))(X-E(X))^T}=E{XX^T}-E{XE(X)^T}-E{(E(X)X^T)}+E{E(X)E(X)^T}。<br>
因为E{XE(X)^T}=E{E(X)X^T}=E(X)E(X)^T<br>
所以C=E{XX^T}-E(X)E(X)^T
