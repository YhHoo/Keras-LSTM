# The source code are retrieved from
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# LSTM with Keras

from sklearn import datasets
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

# visualize the csv dataset
print(series.head())

# split the dataset into training and validation
x = series.values
trainSet, testSet = x[0:-12], x[-12:]

# walk forward validation
history = [x for x in trainSet]
prediction = []
for i in range(len(testSet)):
    # make prediction
    prediction.append(history[-1])
    # make observation
    history.append(testSet[i])

# report performance
rmse = sqrt(mean_squared_error(testSet, prediction))
print('Root Mean Square Error = {:.3f}'.format(rmse))
pyplot.plot(testSet)
pyplot.plot(prediction)
pyplot.show()






# iris = datasets.load_iris()

# attendance = {'Day': [1, 2, 3, 4, 5, 6],
#               'Absent': [0, 0, 1, 2, 3, 4],
#               'Weather': [1, 1, 1, 0, 0, 0]}
#
# df = pd.DataFrame(attendance)
# df.set_index('Day', inplace=True)
# print(df.Weather.tolist())



