# The source code are retrieved from
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# Develop an LSTM forecast model for a one-step univariate time series forecasting problem.

from pandas import read_csv
from pandas import datetime
from math import sqrt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error


def parser(x):
    # this x will receive 1-01, 1-02 ...
    # then join to become 1901-01, and first term is Year,second is month
    return datetime.strptime('190'+x, '%Y-%m')


series = read_csv('shampoo-sales.csv',
                  header=0,
                  parse_dates=[0],  # this tells the pandas to parse the column 0 as date
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)  # then this fn converts the column of string to
                                       # an array of datetime instances

# visualize the first 5 rows of data set
print('Data Overview\n', series.head(), '\n')
# put all data in sales column into a list
x = series.values
# split the data set into training(2/3) and validation(1/3)
trainSet, testSet = x[0:-12], x[-12:]

# ---------------------[Persistence Model Forecast]------------------------------
# The persistence forecast is where the observation from the prior time step (t-1)
# is used to predict the observation at the current time step (t).
# -------------------------------------------------------------------------------

# walk forward validation
history = [x for x in trainSet]

prediction = []
for i in range(len(testSet)):
    # make prediction
    prediction.append(history[-1])
    # make observation
    history.append(testSet[i])

# report performance
# basically prediction[] is exactly same as history[]
# except it is one month ahead. They are just trying to create an
# close prediction manually

rmse = sqrt(mean_squared_error(testSet, prediction))
print('Root Mean Square Error = {:.3f}'.format(rmse))
pyplot.setp(pyplot.plot(testSet), color='r')
pyplot.setp(pyplot.plot(prediction), color='b')
pyplot.show()









