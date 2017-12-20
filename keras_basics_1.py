# The source code are retrieved from
# https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
# It uses Stateful and Stateless LSTM network, tgt with diff kind of inputs training set to train the model
# to

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

# global variables
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


class LstmNetwork:
    # declaring class global fields
    model = Sequential()
    # accessing the global variables
    global alphabet, char_to_int, int_to_char

    # raw_data is the series of alphabets 'ABC...Z'
    # data_x, data_y are list of unscaled inputs and labels
    # data_x_processed and data_y_processed are those processed, ready for supervised training
    # char_to_int and int_to_char are both just dict {'A':0, ..} and {0:'A', ...}
    def __init__(self, data_x, data_y, data_x_processed, data_y_processed):
        self.x = data_x
        self.y = data_y
        self.inputs = data_x_processed
        self.labels = data_y_processed

    # inputs and labels comes in pairs, as a supervised training set
    # nb_epochs is no of training it carries out with the same data set
    def training(self, nb_epochs, batch_size, shuffle, load_model=False):
        self.model.add(LSTM(32,
                            input_shape=(self.inputs.shape[1], self.inputs.shape[2]),
                            stateful=False))
        self.model.add(Dense(self.labels.shape[1], activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # load previously saved model, lets only use ONE model for all stateless training to avoid confusion
        # this feature is to enable user to load the previously saved model and continue training from the there
        # instead of starting again all the way from 0.
        filepath = 'alphabet_best_model_stateless.hdf5'
        if load_model:
            print('Loading previously saved model: {}'.format(filepath))
            self.model.load_weights(filepath)
        # create checkpoint to save best model while training
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     period=5)
        callback_list = [checkpoint]
        # start training and saving best model
        print('Training Started...')
        self.model.fit(self.inputs,
                       self.labels,
                       epochs=nb_epochs,
                       batch_size=batch_size,
                       verbose=2,
                       shuffle=shuffle,
                       callbacks=callback_list)
        print('Training Completed and Best Model Saved!')

    # this training turn on the Stateful, and reset the state after each epoch. It is to show the effect
    # of turning on Stateful for LSTM
    def training_stateful(self, nb_epoch):
        batch_size = 1  # here fix batch size = 1
        # input_shape becomes 3D from 2D, with batch size as first dim,
        self.model.add(LSTM(16,
                            batch_input_shape=(batch_size, self.inputs.shape[1], self.inputs.shape[2]),
                            stateful=True))
        self.model.add(Dense(self.labels.shape[1], activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # it now split the epoch into individual controlled iteration so we can reset state after every epoch
        print('Stateful Training Started...')
        # -----------------[SAVING]------------------
        # checkpoint- this will save the model during training every time the accuracy hits a new highest
        filepath = 'alphabet_best_model_stateful.hdf5'
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',  # for acc, it should b 'max'; for loss, 'min'
                                     period=5)  # no of epoch btw checkpoints
        callback_list = [checkpoint]
        # manually reset the state after each epoch
        for i in range(nb_epoch):
            self.model.fit(self.inputs,
                           self.labels,
                           epochs=1,
                           batch_size=batch_size,
                           verbose=2,
                           shuffle=False,
                           callbacks=callback_list)
            self.model.reset_states()
        print('Training Completed !')

    def test_accuracy(self, stateful=False):
        print('Testing Model Accuracy...')
        if stateful:
            scores = self.model.evaluate(self.inputs, self.labels, batch_size=1, verbose=0)
            self.model.reset_states()
        else:
            scores = self.model.evaluate(self.inputs, self.labels, verbose=0)
        print('Model Accuracy: {:.2f}'.format(scores[1] * 100))

    def predict_all(self):
        print('Testing all inputs: ')
        for unit_input in self.x:
            # convert each set inside x to 3D and scale it
            x = np.reshape(unit_input, (1, self.inputs.shape[1], self.inputs.shape[2]))
            x = x / float(len(alphabet))
            # feed each of them into the trained model
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = int_to_char[index]  # yh: index might not b int
            seq_in = [int_to_char[values] for values in unit_input]
            print(seq_in, '->', result)

    def predict_random(self):
        print('Testing randomly: ')
        for i in range(20):
            # randomize
            unit_input_index = np.random.randint(len(self.x))
            unit_input = self.x[unit_input_index]
            # convert into [samples, time steps, features] fr 2D
            x = np.reshape(unit_input, (1, self.inputs.shape[1], self.inputs.shape[2]))
            x = x / float(len(alphabet))
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[values] for values in unit_input]
            print(seq_in, '->', result)

    def predict_random_starting(self, start):
        print('Testing starts from \'{}\':'.format(start))
        seed = [char_to_int[start]]  # contain in a list
        for i in range(5):
            x = np.reshape(seed, (1, self.inputs.shape[1], self.inputs.shape[2]))
            x = x / float(len(alphabet))
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)  # take the max among the list
            print(int_to_char[seed[0]], '->', int_to_char[index])
            seed = [index]
        self.model.reset_states()

    def predict_variable_length(self):
        for i in range(20):
            pattern_index = np.random.randint(len(self.x))
            pattern = self.x[pattern_index]
            x = pad_sequences([pattern], maxlen=self.inputs.shape[1], dtype='float32')
            x = np.reshape(x, (1, self.inputs.shape[1], 1))
            x = x / float(len(alphabet))
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            print(seq_in, "->", result)


# this creates a data set for n-Char to One-Char Mapping by LSTM
def n_char_to_one_char_data(sequence_length, window):
    # define sequence length, e.g. for =2: 'A','B' for 'C'; for =3: 'A','B','C' for 'D'
    # this sequence is to supervised training the network
    seq_length = sequence_length  # Change this value to have a diff training outcomes***
    data_x, data_y = [], []
    # create a dict of mappings btw every single char in alphabet to index 0-*
    # e.g. {'A':0, 'B':1, ...}
    # create a supervised training set
    for i in range(0, len(alphabet) - seq_length, 1):
        # initialize seq_in = 'A', seq_out = 'B' and so on for seq_len=1
        # initialize seq_in = 'ABC', seq_out = 'D' and so on for seq_len=3
        seq_in = alphabet[i: i + seq_length]
        seq_out = alphabet[i + seq_length]
        # next convert the char(in seq_in & seq_out) to int, e.g.
        # data_x = [[0], [1], ...], data_y = [[1], [2], ...] for seq_len=1
        # data_x = [[0, 1, 2], [2, 3, 4], ...], data_y = [[3], [4], ...] for seq_len=3
        data_x.append([char_to_int[char] for char in seq_in])
        data_y.append(char_to_int[seq_out])
    # reshape X to be [samples, time steps, features]
    if window == 'time_step':
        data_x_processed = np.reshape(data_x, (len(data_x), seq_length, 1))
    elif window == 'feature':
        data_x_processed = np.reshape(data_x, (len(data_x), 1, seq_length))  # ELSE NEEDED !
    # normalize
    data_x_processed = data_x_processed / float(len(alphabet))
    # one hot encode the output variable, meaning convert 3 to 00010...0, 4 to 000010...0, and so on
    # much like a binary no for categorize purpose. Each of the digit above represents single output of a
    # neuron at output layer
    data_y_processed = np_utils.to_categorical(data_y)

    return data_x, data_y, data_x_processed, data_y_processed


# generate a dataset of input output pairs
def variable_char_to_one_char(max_len=5, num_inputs=1000):
    # prevent max_len too big
    if max_len > 25:
        raise ValueError('max_len greater than 25, please lower it!')
    data_x, data_y = [], []
    for i in range(num_inputs):
        start = np.random.randint(len(alphabet)-2)  # start limit is 23
        end = np.random.randint(low=start, high=min(start+max_len, len(alphabet)-1))  # end limit is 25
        # data visualize
        seq_in = alphabet[start:end+1]  # +1 to prevent start=end
        seq_out = alphabet[end+1]
        # print(seq_in, '->', seq_out)
        # get ready for training
        data_x.append([char_to_int[char] for char in seq_in])
        data_y.append(char_to_int[seq_out])
    # pad_sequence will change [10] to [0, 0, 0, 0, 10] if max_len=5
    data_x_processed = pad_sequences(data_x, maxlen=max_len, dtype='float32')
    data_x_processed = np.reshape(data_x_processed, (data_x_processed.shape[0], max_len, 1))
    data_x_processed = data_x_processed / float(len(alphabet))
    data_y_processed = np_utils.to_categorical(data_y)
    return data_x, data_y, data_x_processed, data_y_processed




