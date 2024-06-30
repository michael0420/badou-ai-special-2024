import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 假设我们有一个Pandas DataFrame
df = pd.DataFrame({
    'A': np.random.rand(100) * 100,  # 生成0到100之间的随机数
    'B': np.random.rand(100) * 100
})

# 初始化StandardScaler
scaler = StandardScaler()

# 标准化数据
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 可视化原始数据和标准化数据
fig, axs = plt.subplots(nrows=2, figsize=(10, 6))

# 绘制原始数据
df.plot(kind='hist', bins=30, alpha=0.6, stacked=False, ax=axs[0], title='Original Data')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')

# 绘制标准化数据
df_scaled.plot(kind='hist', bins=30, alpha=0.6, stacked=False, ax=axs[1], title='Standardized Data')
axs[1].set_xlabel('Standardized Value')
axs[1].set_ylabel('Frequency')

# 显示图形
plt.tight_layout()
plt.show()