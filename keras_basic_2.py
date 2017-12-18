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
# -->


# fix random seed for reproducibility
np.random.seed(7)

# initialize training data set
data_x, data_y, data_x_processed, data_y_processed = \
    kb1.n_char_to_one_char_data(sequence_length=1, window='time_step')  # the sequence length

# instantiate the lstm network
lstm_network = kb1.LstmNetwork(data_x, data_y, data_x_processed, data_y_processed)

# ----[CHOOSE ONLY 1 TRAINING]----
# None stateful training
# lstm_network.training(nb_epochs=5000, batch_size=len(data_x), shuffle=True)
# training stateful
lstm_network.training_stateful(nb_epoch=1200)

# test accuracy
lstm_network.test_accuracy(stateful=True)

# prediction visualization
lstm_network.predict_all()

# prediction random
# lstm_network.predict_random()

# 5 prediction at a randomly chosen starting alphabet
lstm_network.predict_random_starting('K')

# Data Visualization before Training, Uncomment to watch if u are confused
print('raw data:\n', kb1.alphabet)
print('Data_x:\n', data_x)
print('Data_y:\n', data_y)
print('Data_x_processed:\n', data_x_processed)
print('Data_y_processed:\n', data_y_processed)




