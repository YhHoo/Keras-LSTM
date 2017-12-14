import keras_basics_1 as kb1
import numpy as np


# fix random seed for reproducibility
np.random.seed(7)

# initialize training data set
raw_data, data_x, data_y, data_x_processed, data_y_processed, char_to_int, int_to_char = \
    kb1.state_within_a_batch_data(1)  # the sequence length


# instantiate the lstm network
lstm_network = kb1.LstmNetwork(raw_data, data_x, data_y, data_x_processed, data_y_processed, char_to_int, int_to_char)
# training
lstm_network.training(nb_epochs=5000, batch_size=len(data_x), shuffle=False)
# test accuracy
lstm_network.test_accuracy()
# prediction visualization
lstm_network.predict_all()
# prediction random
lstm_network.predict_random()


# Data Visualization before Training, Uncomment to watch if u are confused
print('raw data:\n', raw_data)
print('Data_x:\n', data_x)
print('Data_y:\n', data_y)
print('Data_x_processed:\n', data_x_processed)
print('Data_y_processed:\n', data_y_processed)




