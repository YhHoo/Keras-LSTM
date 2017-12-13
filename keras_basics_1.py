import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils


class LstmNetwork:
    # declaring class global fields
    model = Sequential()

    # raw_data is the series of alphabets 'ABC...Z'
    # data_x, data_y are list of unscaled inputs and labels
    # data_x_processed and data_y_processed are those processed, ready for supervised training
    # char_to_int and int_to_char are both just dict {'A':0, ..} and {0:'A', ...}
    def __init__(self, raw_data, data_x, data_y, data_x_processed, data_y_processed, char_to_int, int_to_char):
        self.alphabet = raw_data
        self.x = data_x
        self.y = data_y
        self.inputs = data_x_processed
        self.labels = data_y_processed
        # create a dict of mappings btw every single char in alphabet to index 0-*
        # e.g. {'A':0, 'B':1, ...}
        self.char_map_int = char_to_int
        self.int_map_char = int_to_char

    # inputs and labels comes in pairs, as a supervised training set
    # nb_epochs is no of training it carries out with the same data set
    def training(self, nb_epochs):
        self.model.add(LSTM(32, input_shape=(self.inputs.shape[1], self.inputs.shape[2])))
        self.model.add(Dense(self.labels.shape[1], activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Training Started...')
        self.model.fit(self.inputs, self.labels, epochs=nb_epochs, batch_size=1, verbose=2)
        print('Training Completed !')

    def test_accuracy(self):
        print('Testing Model Accuracy...')
        scores = self.model.evaluate(self.inputs, self.labels, verbose=0)
        print('Model Accuracy: {:.2f}'.format(scores[1] * 100))

    def predict(self):
        for unit_input in self.x:
            x = np.reshape(unit_input, (1, len(unit_input), 1))
            x = x / float(len(self.alphabet))
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = self.int_map_char[index]  # yh: index might not b int
            seq_in = [self.int_map_char[values] for values in unit_input]
            print(seq_in, '->', result)


# this creates a data set for One-Char to One-Char Mapping by Stateful LSTM
def one_char_to_one_char_data():
    # define sequence length, e.g. for =2: 'A','B' for 'C'; for =3: 'A','B','C' for 'D'
    # this sequence is to supervised training the network
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    seq_length = 1
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
        data_x.append([char_map_int[char] for char in seq_in])
        data_y.append([char_map_int[char] for char in seq_out])
    # reshape X to be [samples, time steps, features]
    data_x_processed = np.reshape(data_x, (len(data_x), seq_length, 1))
    # normalize
    data_x_processed = data_x_processed / float(len(alphabet))
    # one hot encode the output variable
    data_y_processed = np_utils.to_categorical(data_y)

    return alphabet, data_x, data_y, data_x_processed, data_y_processed, char_map_int, int_map_char





