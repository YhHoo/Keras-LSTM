# this is one-hot encoding Retrieved from
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# This encoding is a representation of categorical variables in binary vectors
# This is necessary when training an LSTM network to recognize a sequence
import numpy as np
from keras.utils import to_categorical

data = 'hello world'
print('Input: ', data)
# create a maps
alphabet = 'abcdefghijklmnopqrstuvwxyz '
# ----------[Using Manual Hot Encoding]-------------
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# convert the data to integers representation
data_in_int = [char_to_int[char] for char in data]
print('Int. Rep: ', data_in_int)

one_hot_encoded = []

for i in data_in_int:
    # declare an all 0 list [0, 0, ..., 0]
    binary_encoding = [0 for _ in range(len(alphabet))]
    binary_encoding[i] = 1
    # print(binary_encoding)
    one_hot_encoded.append(binary_encoding)

# convert to np array so it prints the binary vector in rows
print('With Manual Hot Encoding: ')
temp = np.asarray(one_hot_encoded)
print(temp)

# -------------[Using Keras Lib]----------------
# Much faster !

encoded = to_categorical(data_in_int)
print('One-Hot encoding wiht Keras lib:\n', encoded)
# invert encoding
decoded = np.argmax(encoded)
print('Keras decoded:\n', decoded)






