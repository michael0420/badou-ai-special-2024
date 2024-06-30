import numpy as np


class PCA:
    def __init__(self, src_data, n_components):
        self.src_data = src_data
        self.n_components = n_components
        self._centrial_data = self._centrialized_data()
        self._convarians_matrix = self.convarians_matrix()
        self._dst_vector = self.eigen()

    def _centrialized_data(self):
        centrial_data = self.src_data - self.src_data.mean(axis=0)
        return centrial_data

    def convarians_matrix(self):
        convarians_matrix = np.dot(self._centrial_data.T, self._centrial_data)/self.src_data.shape[0]
        return convarians_matrix

    def eigen(self):
        eig_value, eig_vector = np.linalg.eig(self._convarians_matrix)
        sort = np.argsort(-eig_value)
        # dst_eig_vector =np.transpose(np.array([eig_vector[i] for i in sort[:self.n_components]]))
        dst_eig_vector = np.transpose([eig_vector[:, sort[i]] for i in range(self.n_components)])
        print(dst_eig_vector)
        return dst_eig_vector

    def dst_data(self):
        return np.dot(self.src_data, self._dst_vector)


data = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
dst_n_components = 2
pca = PCA(data, dst_n_components)
print(pca.dst_data())
