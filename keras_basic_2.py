import keras_basics_1 as kb1
import numpy as np

# ----[Configure the Network]----
# Firstly, the sequence_length, predict_all() & predict_random() are free to vary under any case
# [A] One-Char-to-One-Char Training(Feature Window)
# --> Set window='feature'; batch_size=1; epoch=500
#
# [B] One-Char-to-One-Char Training(Time_step Window)
# --> Set window='time_step'; batch_size=1; epoch=500
#
# [C] All-in-one-Batch Training
# --> Set batch_size=len(data_x); shuffle=False; epoch=5000; window=any
#
# [D] Stateful LSTM Training(this is to train network to be able to match any no. of sequence to 1 alphabet, instead
# of only 1 or 3. It also means the state of one batch are being used on next batch)
# --> use training_stateful , nb_epoch need to b very big because oni one batch for one epoch
# --- But the accuracy is just 90% sth and the predict_random_starting() fails
#
# [E] Non-stateful but using Variable-length-char-to-1-char


# fix random seed for reproducibility
np.random.seed(7)

# initialize training data set
data_x, data_y, data_x_processed, data_y_processed = \
    kb1.variable_char_to_one_char(max_len=5, num_inputs=1000)  # the sequence length

# instantiate the lstm network
lstm_network = kb1.LstmNetwork(data_x, data_y, data_x_processed, data_y_processed)

# ----[CHOOSE ONLY 1 TRAINING]----
# Non stateful training
lstm_network.training(nb_epochs=500, batch_size=len(data_x), shuffle=True, load_model=True)
# training stateful
# lstm_network.training_stateful(nb_epoch=1000)

# test accuracy
# lstm_network.test_accuracy(stateful=False)

# prediction visualization
# lstm_network.predict_all()

# prediction random
# lstm_network.predict_variable_length()

# 5 prediction at a randomly chosen starting alphabet
# lstm_network.predict_random_starting('K')

# Data Visualization before Training, Uncomment to watch if u are confused
# print('raw data:\n', kb1.alphabet)
# print('Data_x:\n', data_x)
# print('Data_y:\n', data_y)
# print('Data_x_processed:\n', data_x_processed)
# print('Data_y_processed:\n', data_y_processed)




