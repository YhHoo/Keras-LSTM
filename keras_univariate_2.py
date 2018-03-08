# this file load the model trained from keras_univariate_1.py
# and feed in the same test set to inspect for correctness of
# prediction by loaded model

from sklearn.metrics import mean_squared_error
from math import sqrt
import keras_univariate_1 as ku1
from matplotlib import pyplot
from keras.models import model_from_json

# ----[Loading Model]----
# load json and create model
json_file = open('shampoo_model.json', 'r')
loaded_json_model = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_json_model)
# load weights into new model, according to doc, weight has to be loaded fr .h5 only
# bcaz json store only the structure of model, h5 store the weights
lstm_model.load_weights('shampoo_model.h5')
print('Model Loaded !')
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# ----[Walk forward Validation]----
predictions = []
for i in range(len(ku1.test_data_scaled)):
    # Input test data into the trained model and store the model prediction in np array yHat
    # But here did it one by one, month by month instead of one shot
    X, y = ku1.test_data_scaled[i, 0:-1], ku1.test_data_scaled[i, -1]  # y is trivial
    yhat = ku1.forecast_lstm(lstm_model, 1, X)
    # uncomment line below and comment the line above to make sure inverse function works correctly
    # we should get RMSE=0
    # yhat = y
    # invert scaling
    yhat = ku1.invert_scale(ku1.scaler, X, yhat)
    # invert differencing
    yhat = ku1.inverse_difference(ku1.raw_values, yhat, len(ku1.test_data_scaled) + 1 - i)
    # store model prediction in a list
    predictions.append(yhat)

# ----[Result Visualization]----
pyplot.plot(predictions, 'r', ku1.raw_values[-12:], 'b')
# Report Performance - test the model prediction of outputs of train data with it's expected output
rmse = sqrt(mean_squared_error(ku1.raw_values[-12:], predictions))
print('%d) Test RMSE: %.3f' % (1, rmse))
pyplot.show()
