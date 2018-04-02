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
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt


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


data = read_csv('air_quality_dataset_processed.csv', index_col=0)
# Dropping Pollution with NaN(0) rows
# pollution_zero_row = data.index[data['pollution'] == 0].tolist()
# data.drop(pollution_zero_row, inplace=True)

data = data[:40003]  # THE ADJUSTER !! FOR DIVISIBLE BATCH_INPUT_SHAPE
print(data.head())
pollution_data = data.values[:, 0].astype(dtype='float32')
pollution_data = np.reshape(pollution_data, (-1, 1))
print('RESHAPED to VERTICAL---------------\n', pollution_data)

# scaling
scaler = MinMaxScaler(feature_range=(0, 1))
pollution_data_scaled = scaler.fit_transform(pollution_data)
print('SCALED---------------\n', pollution_data_scaled[:10])

# supervised
time_step = 3
data_dim = 1
pollution_data_supervised = series_to_supervised(pollution_data_scaled, n_in=time_step, n_out=1)
print('SUPERVISED--------------Size={}\n'.format(pollution_data_supervised.shape), pollution_data_supervised[:10])

# split into train test
split = 0.7
train_size = int(split * pollution_data_supervised.shape[0])
test_size = pollution_data_supervised.shape[0] - train_size
pollution_data_supervised = pollution_data_supervised.values

# slicing
train_X = pollution_data_supervised[:train_size, :-1]
train_y = pollution_data_supervised[:train_size, -1]
test_X = pollution_data_supervised[train_size:, :-1]
test_y = pollution_data_supervised[train_size:, -1]

# convert to 3D
train_X_3d = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], data_dim))
test_X_3d = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], data_dim))

print('---------[READY]------------')
print('TRAIN_X = {}\nTRAIN_y = {}'.format(train_X.shape, train_y.shape))
print('TEST_X  = {}\nTEST_y  = {}'.format(test_X.shape, test_y.shape))
print('TRAIN_X_3D = ', train_X_3d.shape)
print('TEST_X_3D = ', test_X_3d.shape)


# model architecture building -----------------------------------------------
batch_size = 100
epoch = 15

model = Sequential()
model.add(LSTM(100,
               input_shape=(train_X_3d.shape[1], train_X_3d.shape[2]),
               return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
print(model.summary())
# model checkpoint to save model of Lowest VAL_LOSS
filepath = 'air_quality_model.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',  # for acc, it should b 'max'; for loss, 'min'
                             period=1)  # no of epoch btw checkpoints
callback_list = [checkpoint]
history = model.fit(x=train_X_3d,
                    y=train_y,
                    epochs=epoch,
                    batch_size=100,
                    validation_data=(test_X_3d, test_y),
                    verbose=2,
                    callbacks=callback_list,
                    shuffle=True)


# ----[Saving Model Structure ONLY, not weight]----
# serialize and saving the model structure to JSON
model_name = 'air_quality_model'
model_json = model.to_json()
with open(model_name + '.json', 'w') as json_file:
    json_file.write(model_json)
print('Model Structure Saved !')


# ----[VISUALIZE]-----
# Plotting of loss over epoch
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.show()

# ----[PREDICTION AND VERIFICATION]-----------------------------------------------
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

# start prediction
prediction = model.predict(test_X_3d, batch_size=batch_size)
prediction = scaler.inverse_transform(prediction)

test_y = test_y.reshape((-1, 1))
actual = scaler.inverse_transform(test_y)

# calc RMSE
rmse = sqrt(mean_squared_error(actual, prediction))

# visualize
plt.plot(prediction[:72], marker='x', label='PREDICTION')
plt.plot(actual[:72], marker='o', label='ACTUAL')
plt.legend()
plt.title('LSTM PREDICTION vs ACTUAL for 3 Days\n RMSE={:.3f}'.format(rmse))
plt.show()

print('RMSE = ', rmse)