import numpy as np
from scipy.linalg import lstsq
import pandas as pd

sales = pd.read_csv('train_data.csv', engine='python', sep='\s*,\s*')
X = sales['X'].values  # 存csv的第一列
Y = sales['Y'].values  # 存csv的第二列

X_1 = np.reshape(X, (X.shape[0], 1))
Y_1 = np.reshape(Y, (Y.shape[0], 1))

x, residues, rank, s = lstsq(X_1, Y_1)

print(x)
