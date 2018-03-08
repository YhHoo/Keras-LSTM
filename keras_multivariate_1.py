# this code and dataset are retrieved from
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from pandas import read_csv
from pandas import datetime
import numpy as np


# ------------------[DATA PROCESSING]----------------------
def parser(x):
    return datetime.strptime(x, '%Y %m %d %H')


# read_csv() automatically put the data in df
dataset = read_csv('air_quality_dataset.csv',
                   index_col=0,
                   parse_dates=[['year', 'month', 'day', 'hour']],  # accessing 4 columns of datetime
                   date_parser=parser)
# drop the column of 'No'
dataset.drop('No', axis=1, inplace=True)
# Changing the column's name
dataset.index.name = 'Date'  # the index column name
# the rest of the column name
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# mark all NA as 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours, here it means, consider only from row 24 onwards
# The reason of deletion is due to their NA pollution index for whole day
dataset = dataset[24:]
# visualize
print(dataset.head(10))
# save processed data to file
dataset.to_csv('air_quality_dateset_processed.csv')



