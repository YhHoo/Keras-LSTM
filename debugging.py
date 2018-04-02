import numpy as np
from math import sin, radians, sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from functools import reduce
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from pandas import read_csv
from pandas import DataFrame

x = 3
y = 4
assert (x == y), 'Feature setting must be same'


# ------------[EXPERIMENT OF LSTM ON SINE WAVE]-------------
# # sampling rate
# fs = 200
# # frequency of the wave (e.g. 2 for 2 repetition)
# f = 2
# x = np.arange(200)
# y = [sin(2 * np.pi * f * (i / fs)) for i in x]
# data = np.array([x, y])
# data = data.T
# # plt.plot(data[:100, 0], data[:100, 1])
# # plt.title('TRAINING DATA')
# # plt.show()
#
# # prepare TRAINING DATA
# train_x = []
# for i in range(2, 99, 1):
#     temp = []
#     for j in range(2, -1, -1):
#         temp.append(data[i-j, 1])
#     train_x.append(temp)
# train_x = np.array(train_x[:-1])
# train_y = data[3:100, 1]
# train_y = np.array(train_y[:-1])
# train_x_3d = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
#
# # prepara TESTING DATA
# test_x = []
# for i in range(110, 198, 1):
#     temp = []
#     for j in range(2, -1, -1):
#         temp.append(data[i-j, 1])
#     test_x.append(temp)
# test_x = np.array(test_x)
# test_y = data[111:199, 1]
# test_y = np.array(test_y)
# test_x_3d = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
#
# print(train_x_3d.shape)
# print(test_x_3d.shape)
#
#
# # training
# nb_batch = 2
# model = Sequential()
# model.add(LSTM(50,
#                batch_input_shape=(nb_batch, train_x_3d.shape[1], train_x_3d.shape[2]),
#                stateful=False))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
#
# history = model.fit(x=train_x_3d,
#                     y=train_y,
#                     epochs=100,
#                     batch_size=nb_batch,
#                     verbose=2,
#                     validation_data=(test_x_3d, test_y))
# prediction = model.predict(test_x_3d, batch_size=nb_batch)
# plt.plot(test_y, marker='o', label='Actual')
# plt.plot(prediction, marker='x', label='Prediction')
# plt.legend()
# plt.show()
# rmse = sqrt(mean_squared_error(test_y, prediction))
# print('RMSE = ', rmse)


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
    dummy = np.array([[-980, 2, 3, 4, -5],
                      [11, 22, 33, 44, -55],
                      [9, 9, 9, 9, -99],
                      [769, 5, 5, 5, -500]])
    dummy2 = scaler.fit_transform(dummy)
    print(dummy2)
    dummy2 = scaler.inverse_transform(dummy2)
    print(dummy2)
    scaler2 = StandardScaler()
    dummy3 = scaler2.fit_transform(dummy)
    print(dummy3)


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


#
# test = np.array([[1, 2, 3],
#                  [11, 23, 34],
#                  [23, 35, 46],
#                  [33, 44, 55]])
# diff_list = []
# for i in range(test.shape[1]):
#     diff_list.append(difference(test[:, i]))
# diff_list = np.asarray(diff_list)
# diff_list = diff_list.T
# print(diff_list)
#
# normal = inv_difference(head=1, diff_list=diff_list[:, 0])
# print(normal)


# l = np.array([[1, 2, 3, 4]]).reshape((4, 1))
# print(l.shape[0])
# z = np.zeros((4, 2))
# print(z)
# z = np.concatenate((l, z), axis=1)
# # # print(l.shape)
# print(z)

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



