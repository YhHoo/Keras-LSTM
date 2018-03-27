import numpy as np
from math import sin, radians
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from functools import reduce
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from pandas import DataFrame

# df = DataFrame([[1, 2, 3, 4], [11, 22, 33, 44]], columns=['A2', 'B2', 'A1', 'B1'])
# print(df)
# df2 = df.values
#
# # it means, a list first hav 2 big items, each of the 2 hav 3 other items inside,
# # then each of the 3 items has 4 another items inside
# zero = np.zeros((2, 3, 4))
# print(zero)
#
#
# df3 = np.reshape(df2, (2, 2, 2))
# print(df3)


def difference(datalist, interval=1):
    diff = []
    for i in range(interval, len(datalist)):
        diff.append(datalist[i] - datalist[i - interval])
    return diff


def inv_difference(head, diff_list):
    inv_list = []
    accu = head
    inv_list.append(accu)
    for i in range(len(diff_list)):
        accu += diff_list[i]
        inv_list.append(accu)
    return inv_list


test = np.array([[1, 2, 3],
                 [11, 23, 34],
                 [23, 35, 46],
                 [33, 44, 55]])
diff_list = []
for i in range(test.shape[1]):
    diff_list.append(difference(test[:, i]))
diff_list = np.asarray(diff_list)
diff_list = diff_list.T
print(diff_list)

normal = inv_difference(head=1, diff_list=diff_list[:, 0])
print(normal)




# this returns a list of factors of n
def factors(n):
    factor_list = reduce(list.__add__, ([i, n//i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0))
    return factor_list


def fill_up_cavity():
    data_in = [[1, 2, 3], [5, 6, 7, 8], [7, 8], [9]]
    data_in_filled_up = pad_sequences(data_in, maxlen=5, dtype='float32')
    print(data_in_filled_up)


def test_label_encoder():
    l = ['N', 'W', 'S', 'EA', 'E']
    encoder = LabelEncoder()
    l_encode = encoder.fit_transform(l)
    print(l_encode)


def max_min_scale():
    scaler = MinMaxScaler(feature_range=(0, 1))
    dummy = np.array([[1, 2, 3, 4, -5],
                      [11, 22, 33, 44, -55],
                      [9, 9, 9, 9, -99],
                      [5, 5, 5, 5, -500]])
    dummy2 = scaler.fit_transform(dummy)
    print(dummy2)


def test_standard_scaler():
    data = np.array([1, 2, 3, 4, 5]).reshape((5, 1))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print(data_scaled)


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



# sine_wave = [sin(radians(x)) for x in range(360)]
# sine_wave = sine_wave * 2
# plt.plot(sine_wave)
# plt.show()

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



# dummy[1:3, 2:4] = np.array([[7, 7], [7, 7]])
