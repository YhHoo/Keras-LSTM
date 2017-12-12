# The source code are retrieved from
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# Develop an LSTM forecast model for a one-step univariate time series forecasting problem using Keras.

import numpy
import time
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot


def parser(x):
    # this x will receive 1-01, 1-02 ...
    # then join to become 1901-01, and first term is Year,second is month
    return datetime.strptime('190'+x, '%Y-%m')


# create a difference series
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        diff.append(dataset[i] - dataset[i - interval])
    return diff


# invert difference series
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# frame a sequence as a supervised problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    # include the original df
    columns.append(df)
    # join all 3 df side by side
    df = concat(columns, axis=1)
    # fill NaN with zeros
    df.fillna(0, inplace=True)
    return df


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))  # initialize Scaler
    scaler = scaler.fit(train)  # compute the max n min to be used for later scaling
    # transform train
    # train = train.reshape(train.shape[0], train.shape[1])  # yh: useless**
    train_scaled = scaler.transform(train)
    # transform test
    # test = test.reshape(test.shape[0], test.shape[1])  # yh: useless**
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    # reshape into [samples, time steps, features] where LSTM layer expects input to be
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # neurons being the no. of memory blocks.
    # putting a LSTM layer into a keras model and compile
    # batch size defines the number of data are feed into the network during each epoch
    # batch_input_shape is a tuple that defines expected no of observation to read each batch,
    # the no of time step and no of features.
    # then compile the network into efficient symbolic rep using tensorflow as backend math lib.
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # start training the model with train data by iterating for a no. specified by epoch
    for i in range(nb_epoch):
        # epochs is the no. of iteration the model is trained over d entire data provided
        # YH: i tink this 'for loop' is to provide control for us to reset the state everytime an epoch
        # is completed. If we do not wish to reset everytime, we can control iteration no by
        # epochs inside fit()
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print('Epoch {}/{}'.format(i, nb_epoch))
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# just to print out step by step to watch the changes
def yh_data_visualize():
    print('Data Overview\n', series.head(), '\n')
    print('Supervised_values(with shape: {}):\n'.format(supervised_values.shape), supervised_values, '\n')
    print('Training Data Scaled(with shape: {}):\n'.format(train_data_scaled.shape), train_data_scaled, '\n')
    # print('Test Data Scaled:\n', test_data_scaled, '\n')



def read_fr_csv(filename):
    return read_csv(filename,
                    header=0,
                    parse_dates=[0],  # this tells the pandas to parse the column 0 as date
                    index_col=0,
                    squeeze=True,
                    date_parser=parser)  # then this fn converts the column of string to
                                         # an array of datetime instances


# read from csv
series = read_fr_csv('shampoo-sales.csv')

# transform time series to stationary
raw_values = series.values  # put only value in a list
diff_values = difference(raw_values, interval=1)

# transform data to supervised learning
supervised = timeseries_to_supervised(diff_values, lag=1)
supervised_values = supervised.values

# split data into train and test
# note that exact index is not used, instead, -12 means the 12th index from the last
train_data, test_data = supervised_values[0:-12], supervised_values[-12:]

# scale the data
scaler, train_data_scaled, test_data_scaled = scale(train_data, test_data)


# this is to prevent the code fr here onwards is executed when
# this file is imported as a module
if __name__ == '__main__':
    # repeat the same training to find the best
    repeats = 1
    error_scores = []
    for r in range(repeats):
        # timer
        start = time.clock()

        # -----[Training of Model]---- with TRAIN data set
        # train the model for epochs-times(changeable), returned a trained model
        lstm_model = fit_lstm(train_data_scaled, 1, 500, 4)
        # forecast the entire training data set to build up state for forecasting
        # change the input array fr 2d to 3d
        train_data_reshaped = train_data_scaled[:, 0].reshape(len(train_data_scaled), 1, 1)
        lstm_model.predict(train_data_reshaped, batch_size=1)
        print('Training {}/{} completed'.format(r+1, repeats))

        # -----[Walk Forward Validation]---- with TEST data set
        predictions = []
        for i in range(len(test_data_scaled)):
            # Input test data into the trained model and store the model prediction in np array yHat
            # But here did it one by one, month by month instead of one shot
            X, y = test_data_scaled[i, 0:-1], test_data_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_data_scaled) + 1 - i)
            # store model prediction in a list
            predictions.append(yhat)

        # visualize the raw test data with the predicted data
        pyplot.plot(predictions, 'r', raw_values[-12:], 'b')
        # Report Performance - test the model prediction of outputs of train data with it's expected output
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r + 1, rmse))
        print('Time Taken: {:.3f}'.format(time.clock() - start), 's\n')
        error_scores.append(rmse)
        lstm_model.save('shampoo_model_1.h5')

    # summarize results in box plot
    # results = DataFrame()
    # results['rmse'] = error_scores
    # print(results.describe())
    # results.boxplot()
    pyplot.show()









# ---------------------[Persistence Model Forecast]------------------------------
# The persistence forecast is where the observation from the prior time step (t-1)
# is used to predict the observation at the current time step (t).
# -------------------------------------------------------------------------------

# # put all data in sales column into a list
# x = series.values
# # split the data set into training(2/3) and validation(1/3)
# trainSet, testSet = x[0:-12], x[-12:]

# # walk forward validation
# history = [x for x in trainSet]
#
# prediction = []
# for i in range(len(testSet)):
#     # make prediction
#     prediction.append(history[-1])
#     # make observation
#     history.append(testSet[i])
#
# # report performance
# # basically prediction[] is exactly same as history[]
# # except it is one month ahead. They are just trying to create an
# # close prediction manually
#
# rmse = sqrt(mean_squared_error(testSet, prediction))
# print('Root Mean Square Error = {:.3f}'.format(rmse))
# pyplot.setp(pyplot.plot(testSet), color='r')
# pyplot.setp(pyplot.plot(prediction), color='b')
# pyplot.show()









