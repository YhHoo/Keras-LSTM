import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

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
    def training(self, nb_epochs, batch_size, shuffle):
        self.model.add(LSTM(32,
                            input_shape=(self.inputs.shape[1], self.inputs.shape[2]),
                            stateful=False))
        self.model.add(Dense(self.labels.shape[1], activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Training Started...')
        self.model.fit(self.inputs,
                       self.labels,
                       epochs=nb_epochs,
                       batch_size=batch_size,
                       verbose=2,
                       shuffle=shuffle)
        print('Training Completed !')

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
        for i in range(nb_epoch):
            self.model.fit(self.inputs,
                           self.labels,
                           epochs=1,
                           batch_size=batch_size,
                           verbose=2,
                           shuffle=False)
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
            index = np.argmax(prediction)
            print(int_to_char[seed[0]], '->', int_to_char[index])
            seed = [index]
        self.model.reset_states()


# this creates a data set for n-Char to One-Char Mapping by LSTM
def n_char_to_one_char_data(sequence_length, window):
    # define sequence length, e.g. for =2: 'A','B' for 'C'; for =3: 'A','B','C' for 'D'
    # this sequence is to supervised training the network
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
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


def state_within_a_batch_data(sequence_length):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    seq_length = sequence_length
    data_x, data_y = [], []
    # create a dict of mappings btw every single char in alphabet to index 0-*
    # e.g. {'A':0, 'B':1, ...}
    char_map_int = dict((c, i) for i, c in enumerate(alphabet))
    int_map_char = dict((i, c) for i, c in enumerate(alphabet))
    # mapping of 'A' to 0, ....
    for i in range(0, len(alphabet) - seq_length, 1):
        # extract section of the alphabet and store
        seq_in = alphabet[i: i + seq_length]
        seq_out = alphabet[i + seq_length]
        # save another copy of int rep of the alphabets
        # [] is used to contain all seq in one single index of the array
        data_x.append([char_map_int[char] for char in seq_in])
        data_y.append([char_map_int[char] for char in seq_out])

    # convert list of lists to array and pad sequences if needed
    x = pad_sequences(data_x, maxlen=seq_length, dtype='float32')
    # reshape X to be [samples, time steps, features]
    x = np.reshape(x, (x.shape[0], seq_length, 1))
    # normalize
    data_x_processed = x / float(len(alphabet))
    # neuron at output layer
    data_y_processed = np_utils.to_categorical(data_y)

    return alphabet, data_x, data_y, data_x_processed, data_y_processed, char_map_int, int_map_char

