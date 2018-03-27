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


# this converts the data in 'air_quality_dataset_processed.csv' into data ready for
# supervised training. The PRINT line is to let us have a look on how the data are being
# processed and what it ends up as.
def prepare_data(n_in=1, n_out=1, train_split=0.6):
    data = read_csv('air_quality_dataset_processed.csv', index_col=0)
    # so that when time step is 3, the sample size is 40000, and the batch size cn be set easily
    data = data[:40003]
    # encode dir into integers, e.g. E->1, SE->2 ...
    encoder = LabelEncoder()
    data.iloc[:, 4] = encoder.fit_transform(data['wnd_dir'][:])
    # All FEATURE = [pollution | dew  | temp |  press | wnd_dir | wnd_spd | snow | rain]
    # drop off unwanted columns features
    features_to_drop = ['press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    data.drop(features_to_drop, inplace=True, axis=1)
    # get a matrix copy of all values in the df and convert them to float for scaling
    data_values = data.values.astype(dtype='float32')

    # -----------[SCALING]-----------
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_values = scaler.fit_transform(data_values)
    print('B4 PROCESSED----------------\n', data.head())
    print('AFTER PROCESSED-------------\n', data_values[:5])

    # -----------[PREPARE FOR SUPERVISED TRAINING]-----------
    data_supervised = series_to_supervised(data_values, n_in=n_in, n_out=n_out)
    print('SUPERVISED------------------\n', data_supervised.head())
    # dropped the columns so that the var at t left only the one we wan to forecast
    data_supervised.drop(['var2(t)', 'var3(t)'], inplace=True, axis=1)
    samples_size = data_supervised.shape[0]
    print('SUPERVISED DROPPED-----------Size={}\n{}'.format(samples_size, data_supervised.head()))

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
    return data_train_X, data_train_y, data_test_X, data_test_y


time_step = 3
train_X, train_y, test_X, test_y = prepare_data(n_in=time_step, n_out=1, train_split=0.7)
train_X_3d = np.reshape(train_X, (train_X.shape[0], time_step, int(train_X.shape[1] / time_step)))
test_X_3d = np.reshape(test_X, (test_X.shape[0], time_step, int(test_X.shape[1] / time_step)))
print(train_X_3d.shape)
print(test_X_3d.shape)

# ------------------[TRAINING AND VALIDATION]----------------------
# Available batch size = [1, 26278, 2, 13139, 7, 3754, 14, 1877]
# nb_epoch = 3754
batch_size = 500
# history = []
#
model = Sequential()
model.add(LSTM(32,
               batch_input_shape=(batch_size, train_X_3d.shape[1], train_X_3d.shape[2]),
               return_sequences=False,
               stateful=False))
# model.add(LSTM(32,
#                return_sequences=True,
#                stateful=True))
# model.add(LSTM(32,
#                stateful=True))
model.add(Dense(1))
model.compile(loss='mean_absolute_error',
              optimizer='adam')
# for i in range(nb_epoch):
history = model.fit(x=train_X_3d,
                    y=train_y,
                    validation_data=(test_X_3d, test_y),
                    epochs=250,
                    batch_size=batch_size,  # no of samples per gradient update
                    verbose=2,
                    shuffle=False)
#     model.reset_states()
# # model.reset_states()
# Plotting of loss over epoch
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.show()


# ------------------[RMSE EVALUATION]----------------------
# one_year_hour = 365*24
# x = all_x[one_year_hour:, :]
# x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
# prediction = model.predict(x)
# # Prepare a matrix that has same no of columns as the matrix during scaler.fit_transform
# prediction = np.concatenate((prediction, all_x[one_year_hour:, 1:]), axis=1)
# zeros = np.zeros((prediction.shape[0], (transform_rows-prediction.shape[1])))
# prediction = np.concatenate((prediction, zeros), axis=1)
# prediction = scaler.inverse_transform(prediction)
# prediction = prediction[:, 0]  # PREDICTION done
# prediction = prediction.reshape((prediction.shape[0], 1))
#
# # prepare for actual
# actual = all_y[one_year_hour:]
# actual = actual.reshape((actual.shape[0], 1))
# zeros = np.zeros((actual.shape[0], (transform_rows-actual.shape[1])))
# actual = np.concatenate((actual, zeros), axis=1)
# actual = scaler.inverse_transform(actual)
# actual = actual[:, 0]
# actual = actual.reshape((actual.shape[0], 1))

# calculate rmse (since we are calc rmse, we hv to make sure our loss is mean absolute error
# instead of mean square error, otherwise the training optimizes the mse, and rmse will be bigger
# in the end)
# rmse = sqrt(mean_squared_error(actual, prediction))
# print('RMSE = ', rmse)

