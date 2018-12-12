'''class FillNan(object):
    def __init__(self, matrix, matrix_train, mean, std):
        self.matrix = matrix
        self.matrix_train = matrix_train
        self.mean = mean
        self.std = std
        self.Np = self.matrix.shape[0]
        self.Nf = self.matrix.shape[1]

    def fill_all_nan(self):
        j = 0
        for array in self.matrix:
            i = 0
            j += 1
            F0 = []
            for i in range(self.matrix.shape[1]):
                if np.isnan(array[i]):
                    F0 = F0.append[i]
            for i in range(self.matrix.shape[1]):
                if np.isnan(array[i]):
                    y_train = self.matrix_train[:, F0]
                    X_train = np.delete(X_norm, F0, 1)
                    y_train = y_train.reshape(y_train.shape[0], 1)
                    ridge = SolveRidge(y_train, X_train)
                    w = ridge.run(lamb=10)
                    tmp_mean = np.copy(np.delete(self.mean, F0))
                    tmp_std = np.copy(np.delete(self.std, F0))
                    array_train = (array[~np.isnan(array)] - tmp_mean) / tmp_std
                    y_hat = np.dot(array_train, w) * std[0, F0] + mean[0, F0]
                    array[i] = y_hat
                    self.matrix[j] = array
                    i = len(self.matrix.shape[1])
        return self.matrix'''
