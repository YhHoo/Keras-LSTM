import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler

# 1 sample, 10 time steps, and 1 feature


def difference(datalist, interval=1):
    diff_list = []
    for i in range(interval, len(datalist)):
        diff_list.append(datalist[i] - datalist[i-interval])
    return diff_list


def inverse_difference(diff_list, datalist):
    inv_list = []
    for i in range(len(diff_list)):
        inv_list.append(datalist[i] + diff_list[i])
    return inv_list


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data2 = difference(data)
print(data2)
data3 = inverse_difference(data2, data)
print(data3)

# data2 = np.array([[11, 22, 33, 44, 55, 66, 77, 88, 99, 110]])
# data3 = np.concatenate((data, data2), axis=0)
# data4 = data3.T
# # data2 = np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]])
# data_3d = data4.reshape((2, 5, 2))
# print(data4)
# print(data4.shape)
# print(data_3d)
# print(data_3d.shape)
#
# model = Sequential()
# model.add(LSTM(50, input_shape=(2, 5), return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.fit(X, y, epochs=100, batch_size=10, verbose=2)
# algorithm = SGD(lr=0.3, momentum=0.3)
# model.compile(optimizer=algorithm, loss='mean_squared_error', metrics=['accuracy'])


# scaler = MinMaxScaler(feature_range=(0, 1))
#
dummy = np.array([[1, 2, 3, 4, 5],
                  [11, 22, 33, 44, 55],
                  [9, 9, 9, 9, 99],
                  [5, 5, 5, 5, 500]])
# dummy[1:3, 2:4] = np.array([[7, 7], [7, 7]])
