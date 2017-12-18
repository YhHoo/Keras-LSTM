import keras_basics_1 as kb1
import numpy as np


# fix random seed for reproducibility
np.random.seed(7)

# initialize training data set
data_x, data_y, data_x_processed, data_y_processed = \
    kb1.n_char_to_one_char_data(sequence_length=3, window='feature')  # the sequence length

# instantiate the lstm network
lstm_network = kb1.LstmNetwork(data_x, data_y, data_x_processed, data_y_processed)
# training
lstm_network.training(nb_epochs=500, batch_size=1, shuffle=True)
# test accuracy
lstm_network.test_accuracy()
# prediction visualization
lstm_network.predict_all()
# prediction random
lstm_network.predict_random()


# Data Visualization before Training, Uncomment to watch if u are confused
print('raw data:\n', kb1.alphabet)
print('Data_x:\n', data_x)
print('Data_y:\n', data_y)
print('Data_x_processed:\n', data_x_processed)
print('Data_y_processed:\n', data_y_processed)




