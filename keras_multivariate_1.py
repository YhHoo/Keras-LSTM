# this code and dataset are retrieved from
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt

# ------------------[DATA PROCESSING]----------------------
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


# ------------------[SERIES TO SUPERVISED FUNCTION]----------------------
'''
Frame a time series as a supervised learning dataset.
Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
Returns:
        Pandas DataFrame of series framed for supervised learning.
'''


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # check whether is list or np matrix
    n_vars = 1 if type() is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
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
