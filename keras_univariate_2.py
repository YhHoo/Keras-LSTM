# this file load the model trained from keras_univariate_1.py

from sklearn.metrics import mean_squared_error
from keras.models import load_model
from math import sqrt
import keras_univariate_1 as ku1


lstm_model = load_model('shampoo_model_1.h5')
series = ku1.read_fr_csv('shampoo-sales.csv')

# transform time series to stationary
raw_values = series.values  # put only value in a list
diff_values = ku1.difference(raw_values, interval=1)

# transform data to supervised learning
supervised = ku1.timeseries_to_supervised(diff_values, lag=1)
supervised_values = supervised.values

# split data into train and test
# note that exact index is not used, instead, -12 means the 12th index from the last
train_data, test_data = supervised_values[0:-12], supervised_values[-12:]

# scale the data
scaler, train_data_scaled, test_data_scaled = ku1.scale(train_data, test_data)

error_scores = []
predictions = []
for i in range(len(test_data_scaled)):
    # Input test data into the trained model and store the model prediction in np array yHat
    # But here did it one by one, month by month instead of one shot
    X, y = test_data_scaled[i, 0:-1], test_data_scaled[i, -1]
    yhat = ku1.forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = ku1.invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = ku1.inverse_difference(raw_values, yhat, len(test_data_scaled) + 1 - i)
    # store model prediction in a list
    predictions.append(yhat)


# Report Performance - test the model prediction of outputs of train data with it's expected output
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('%d) Test RMSE: %.3f' % (1, rmse))
error_scores.append(rmse)
