import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1 sample, 10 time steps, and 1 feature

data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
data2 = np.array([[11, 22, 33, 44, 55, 66, 77, 88, 99, 110]])
data3 = np.concatenate((data, data2), axis=0)
data4 = data3.T
# data2 = np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]])
data_3d = data4.reshape((2, 5, 2))
print(data4)
print(data4.shape)
print(data_3d)
print(data_3d.shape)

# scaler = MinMaxScaler(feature_range=(0, 1))
#
# dummy = np.array([[1, 2, 3, 4, 5],
#                   [11, 22, 33, 44, 55],
#                   [9, 9, 9, 9, 9],
#                   [5, 5, 5, 5, 5]])
# dummy[1:3, 2:4] = np.array([[7, 7], [7, 7]])


# # dummy = dummy.astype(dtype='float32')
# dummy_scaled = scaler.fit_transform(dummy)
# print('SCALED:\n', dummy_scaled)
# dummy_del = np.delete(dummy_scaled, 4, axis=1)
# print('COL DELETED:\n', dummy_del)
# # dummy_unscaled = scaler.inverse_transform(dummy_del)
# # print('INVERSED:\n', dummy_unscaled)
#
# mat1 = np.array([[1, 2], [3, 4]])
# mat2 = np.array([[8, 9]])
# mat3 = np.concatenate((mat1, mat2), axis=0)
# print('CONCAT: \n', mat3)
# one_year = 365*24
# zeros = np.zeros((one_year, 1))
# ones = np.ones((100, 1))
# mat = np.concatenate((zeros, ones), axis=0)
# print('zero_dim = {} \n mat_dim = {}'.format(zeros.shape, mat.shape))