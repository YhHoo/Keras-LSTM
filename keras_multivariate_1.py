# this code and dataset are retrieved from
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# AIM: frame the supervised learning problem as predicting the pollution at the current hour (t)
# given the pollution measurement and weather conditions at the prior time step
from math import sqrt
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from functools import reduce
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, RNN
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

# ------------------[STEP 1: DATA PROCESSING]----------------------
# def parser(x):
#     return datetime.strptime(x, '%Y %m %d %H')
#
#
# # read_csv() automatically put the data in df
# dataset = read_csv('air_quality_dataset.csv',
#                    index_col=0,
#                    parse_dates=[['year', 'month', 'day', 'hour']],  # accessing 4 columns of datetime
#                    date_parser=parser)
# # drop the column of 'No'
# dataset.drop('No', axis=1, inplace=True)
# # Changing the column's name
# dataset.index.name = 'Date'  # the index column name
# # the rest of the column name
# dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# # mark all NA as 0
# dataset['pollution'].fillna(0, inplace=True)
# # drop the first 24 hours, here it means, consider only from row 24 onwards
# # The reason of deletion is due to their NA pollution index for whole day
# dataset = dataset[24:]
# # visualize
# print(dataset.head(10))
# # save processed data to file
# dataset.to_csv('air_quality_dataset_processed.csv')

# ------------------[STEP 2: DATA LOADING]----------------------
# load the processed data into df
dataset = read_csv('air_quality_dataset_processed.csv', index_col=0)

# ------------------[DATA VISUALIZE]----------------------
# # extract only the data in a matrix
# dataset_values = dataset.values
# # # specify the position of the plot in subplot
# subplot_index = 1
# # # select only the column index that we want to plot
# groups = [0, 1, 2, 3, 4, 5, 6, 7]
# # creating 7 subplots
# for group in groups:
#     plt.subplot(len(groups), 1, subplot_index)
#     plt.plot(dataset_values[:, group])
#     plt.title(dataset.columns[group], y=0.5, loc='right')  # take column's name
#     subplot_index += 1
# plt.show()


# ------------------[SERIES-TO-SUPERVISED FUNCTION]----------------------
'''
Frame a time series as a supervised learning dataset.
Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
Returns:
        Pandas DataFrame of series framed for supervised learning.        
YH:
        Basically, for E.G. for a list = [1,2,3,4,5], if we set n_in=3, --> Input=[1, 2, 3]; 
        n_out=1, --> [4], and so on. If n_in=2, --> Input=[1, 2] ; n_out=2 --> Output=[2, 3].
        This is called multi-step forecast. 
        Note that input and out means supervised training data pair.
        When features>=2(multi-variate forecast), it will becomes n_in columns match to another n_out columns.
        e.g. when we pass a matrix in, [[1, 10],[2, 20],[3, 30]], it will treat '1,2,3' as same feature in 
        same column and same for '10, 20, 30'
'''


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # check whether is list or np matrix
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1) , n_in will decide how many terms in the sequence
    # n_in means the no of items to be taken as training input
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n), likewise, n_out decide no. of terms in the seq
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


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
        # round off float diff to 4.d.p only
        accu += diff_list[i]
        inv_list.append(accu)
    return inv_list


# this converts the data in 'air_quality_dataset_processed.csv' into data ready for
# supervised training. The PRINT line is to let us have a look on how the data are being
# processed and what it ends up as.
# REMINDER: change those with # ***
def prepare_data(n_in=1, n_out=1, train_split=0.6):
    data = read_csv('air_quality_dataset_processed.csv', index_col=0)
    # This slice the sample size to divisible by 100
    data = data[:40003]
    # encode dir into integers, e.g. E->1, SE->2 ...
    encoder = LabelEncoder()
    data.iloc[:, 4] = encoder.fit_transform(data['wnd_dir'][:])
    # All FEATURE = [pollution | dew  | temp |  press | wnd_dir | wnd_spd | snow | rain]
    # drop off unwanted columns features
    features_to_drop = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    data.drop(features_to_drop, inplace=True, axis=1)
    # get a matrix copy of all values in the df and convert them to float for scaling
    data_values = data.values.astype(dtype='float32')
    print('ORIGINAL----------------\n', data.head())

    # -----------[STATIONARY by DIFFERENCING]-----------
    diff_list = []
    # for every feature column
    for i in range(data_values.shape[1]):
        diff_list.append(difference(data_values[:, i], interval=1))
    # convert back to correct position in matrix
    diff_list = np.asarray(diff_list)
    diff_list = diff_list.T
    print('DIFFERENCE--------------\n', diff_list[:5])

    # -----------[SCALING]-----------
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_values = scaler.fit_transform(diff_list)
    # DEBUGGING
    temp = np.concatenate((diff_list, data_values), axis=1)
    temp = DataFrame(temp)
    temp.to_csv('DEBUGGING SCALER.csv')
    print('SCALED------------------\n', data_values[:5])

    # -----------[PREPARE FOR SUPERVISED TRAINING]-----------
    data_supervised = series_to_supervised(data_values, n_in=n_in, n_out=n_out)
    print('SUPERVISED------------------\n', data_supervised.head())
    # dropped the columns so that the var at t left only the one we wan to forecast
    # data_supervised.drop(['var2(t)', 'var3(t)'], inplace=True, axis=1)
    samples_size = data_supervised.shape[0]
    print('SUPERVISED DROPPED-----------Final_Size={}(Divisible by 100 !)\n{}'.format(samples_size,
                                                                                      data_supervised.head()))

    # -----------[SPLIT INTO TRAINING & TESTING SET]-----------
    train_size = round(samples_size * train_split)
    # get an matrix copy of all values from the df
    data_supervised_values = data_supervised.values
    # slicing
    data_train_X = data_supervised_values[:train_size, :-1]
    data_train_y = data_supervised_values[:train_size, -1]
    data_test_X = data_supervised_values[train_size:, :-1]
    data_test_y = data_supervised_values[train_size:, -1]
    print('---------[READY]------------')
    print('TRAIN_X = {}\nTRAIN_y = {}'.format(data_train_X.shape, data_train_y.shape))
    print('TEST_X  = {}\nTEST_y  = {}'.format(data_test_X.shape, data_test_y.shape))
    return data_train_X, data_train_y, data_test_X, data_test_y, scaler, data.values


time_step = 3
train_X, train_y, test_X, test_y, scaler, data_values_all = prepare_data(n_in=time_step,
                                                                         n_out=1,
                                                                         train_split=0.7)
train_X_3d = np.reshape(train_X, (train_X.shape[0], time_step, int(train_X.shape[1] / time_step)))
test_X_3d = np.reshape(test_X, (test_X.shape[0], time_step, int(test_X.shape[1] / time_step)))
print('TRAIN_X_3D = ', train_X_3d.shape)
print('TEST_X_3D = ', test_X_3d.shape)

# ------------------[TRAINING AND VALIDATION]----------------------
# Available batch size = [1, 26278, 2, 13139, 7, 3754, 14, 1877]
batch_size = 100
# # nb_epoch = 150
# # history = []
# #
# model = Sequential()
# model.add(LSTM(100,
#                input_shape=(train_X_3d.shape[1], train_X_3d.shape[2]),
#                return_sequences=False,
#                stateful=False,
#                dropout=0))
# # model.add(LSTM(32,
# #                return_sequences=False,
# #                stateful=False))
# # model.add(LSTM(32,
# #                stateful=False))
# model.add(Dense(1))
# model.compile(loss='mean_absolute_error',
#               optimizer='adam')
# print(model.summary())
# # for i in range(nb_epoch):
# history = model.fit(x=train_X_3d,
#                     y=train_y,
#                     validation_data=(test_X_3d, test_y),
#                     epochs=20,
#                     batch_size=batch_size,  # no of samples per gradient update
#                     verbose=2,
#                     shuffle=False)
# # model.reset_states()
#
# # ----[Saving Model]----
# # serialize and saving the model structure to JSON
# model_name = 'air_quality_model'
# model_json = model.to_json()
# with open(model_name + '.json', 'w') as json_file:
#     json_file.write(model_json)
# # serialize and save the model weights to HDF5
# model.save_weights(model_name + '.h5')
# print('Model saved !')
#
# # ----[VISUALIZE]-----
# # Plotting of loss over epoch
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='test_loss')
# plt.legend()
# plt.show()


# ------------------[RMSE EVALUATION]----------------------
# ----[Loading Model]----
# load json and create model
json_file = open('air_quality_model.json', 'r')
loaded_json_model = json_file.read()
json_file.close()
model = model_from_json(loaded_json_model)
# load weights into new model, according to doc, weight has to be loaded fr .h5 only
# bcaz json store only the structure of model, h5 store the weights
model.load_weights('air_quality_model.h5')
model.compile(loss='mean_absolute_error',
              optimizer='adam')
print('Model Loaded !')

# ----[Prepare Prediction]----
# inverse transform the prediction back to original values
prediction = []
for i in range(test_X_3d.shape[0]):
    temp_in = np.reshape(test_X_3d[i], (1, 3, 1))
    prediction.append(model.predict(temp_in, batch_size=batch_size)[0])
print(prediction)
# prediction = model.predict(test_X_3d, batch_size=batch_size)
# # prepare zeros matrix so concat with prediction for inverse scaler
# zero = np.zeros((prediction.shape[0], 2))
# # jz to fill up the empty columns
# prediction = np.concatenate((prediction, zero), axis=1)
# inverse MaxMinScale, make sure features are aligned as tat during fit_transform()
prediction = scaler.inverse_transform(prediction)
# # take only first column since the rest are just dummy
# prediction = prediction[:, 0]  # stationary
plt.plot(prediction)
plt.title('PREDICTION IN DIFFERENCE VALUE')
plt.show()
# find the head index, refer evernote for more explanation on below:
head_index = train_X.shape[0] + time_step
# inverse difference
prediction = inv_difference(head=data_values_all[head_index, 0], diff_list=prediction[:, 0])
print('INV_DIFFERENCE------------------\n', prediction[:5])
print(len(prediction))

# ----[Prepare Actual]----
actual = data_values_all[head_index:, 0]
print(len(actual))

# ----[RMSE]----
rmse = sqrt(mean_squared_error(actual, prediction))
print('RMSE = ', rmse)


plt.plot(actual[:72], label='actual', marker='x')
plt.plot(prediction[:72], label='prediction', marker='o')
plt.title('2 DAYS PREDICTION (T_step={}, RMSE={:.3f})'.format(time_step, rmse), )
plt.legend()
plt.show()


