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
# # specify the position of the plot in subplot
# subplot_index = 1
# # select only the column index that we want to plot
# groups = [0, 1, 2, 3, 5, 6, 7]
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


# ---[DEBUG]----
# data_mat = np.array([[1, 2], [10, 20], [100, 200], [1000, 2000]])
# test = series_to_supervised(data_mat, n_in=1, n_out=1, dropnan=True)
# print(test) .

# ------------------[DATA PROCESSING PART 2]----------------------
# HERE WE AIM TO NORMALIZE ALL VALUES TO WITHIN 0-1 WITH DTYPE OF FLOAT32
# get a copy of only the values into a matrix
data_values = dataset.values
# encode the wind dir(at column 5) into integers categories, e.g. SE->1, E->2 ...
encoder = LabelEncoder()
data_values[:, 4] = encoder.fit_transform(data_values[:, 4])
# copy the array and cast to a 'float32'
data_values = data_values.astype(dtype='float32')
# normalize each features values (which are values within columns) in matrix to range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_values = scaler.fit_transform(data_values)

# ------------------[PREPARE FOR SUPERVISED TRAINING SET]----------------------
# create supervised training data in df
reframed_data_values = series_to_supervised(scaled_data_values, n_in=1, n_out=1)
# drop column we dont want to predict
reframed_data_values.drop(reframed_data_values.columns[[9, 10, 11, 12, 13, 14, 15]],  # this return index of column
                          axis=1,
                          inplace=True)
# now prepare training and test sets, take first year for training, the rest for testing
n_train_hours = 365 * 24  # total hours/rows in one year
train_set = reframed_data_values.values[:n_train_hours, :]
test_set = reframed_data_values.values[n_train_hours:, :]
# split into input and output
train_x, train_y = train_set[:, :-1], train_set[:, -1]
test_x, test_y = test_set[:, :-1], test_set[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))  # (8760x8)->(8760x1x8)
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# ------------------[TRAINING AND VALIDATION]----------------------
model = Sequential()
model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))  # input_shape = (1,8)
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
history = model.fit(x=train_x,
                    y=train_y,
                    epochs=50,
                    batch_size=72,
                    validation_data=(test_x, test_y),
                    verbose=2,
                    shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat = model.predict(test_x)
# change back to 2D
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
# invert scaling for FORECAST
# note that we replace the first column of test_x as yhat before pass to inverse transform
# this is because both of them are pollution, and the scaler requires that dim of matrix during
# fit_transform and inverse_transform mz b same !
inv_yhat = np.concatenate((yhat, test_x[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]  # take only first column
# invert scaling for ACTUAL
test_y = test_y.reshape((len(test_y), 1))  # b4 that jz a list actually, now make it array of Nx1
inv_y = np.concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]  # take only first column
# manual calc of RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('RMSE: %.3f' % rmse)
