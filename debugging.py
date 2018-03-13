import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

dummy = np.array([[1, 2, 3, 4, 5], [11, 22, 33, 44, 55], [9, 9, 9, 9, 9], [5, 5, 5, 5, 5]])
dummy = dummy.astype(dtype='float32')
dummy_scaled = scaler.fit_transform(dummy)
print(dummy_scaled)
dummy_del = np.delete(dummy_scaled, 4, axis=1)
print(dummy_del)
dummy_unscaled = scaler.inverse_transform(dummy_del)
print(dummy_unscaled)

# mat3 = np.concatenate((dummy, mat2), axis=1)
