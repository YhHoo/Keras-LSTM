import keras_basics_1 as kb1
import numpy as np


# fix random seed for reproducibility
np.random.seed(7)

# initialize training data set
raw_data, data_x, data_y, data_x_processed, data_y_processed, char_to_int, int_to_char = kb1.one_char_to_one_char_data()
# instantiate the lstm network
lstm_network = kb1.LstmNetwork(raw_data, data_x, data_y, data_x_processed, data_y_processed, char_to_int, int_to_char)
# training
lstm_network.training(nb_epochs=500)
# test accuracy
lstm_network.test_accuracy()
# prediction visualization
lstm_network.predict()








