
#import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Input, Conv1D, Dense
from keras.layers import Dropout, Activation, BatchNormalization
from keras import optimizers
from keras.utils import plot_model
from keras import losses
from keras.preprocessing.sequence import pad_sequences
import tensorflow
from keras import backend
config = tensorflow.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.visible_device_list = "0"
sess = tensorflow.Session(config=config)
backend.set_session(sess)

import numpy as np
from numpy import array as arr
from numpy import argmax

#data
x_traindata_regarray = "./data/x_train_unpad"
y_traindata_regarray = "./data/y_train"
x_valdata_regarray = "./data/x_val_unpad"
y_valdata_regarray = "./data/y_val"
x_testdata_regarray = "./data/x_test_unpad"
y_testdata_regarray = "./data/y_test"

def encode_array(filename):
        f = open(filename, "r")
        array = []
        for line in f:
                data_line = line.split()
                array.append(data_line)
        return array

x_train = encode_array(x_traindata_regarray)
y_train = encode_array(y_traindata_regarray)
x_test = encode_array(x_testdata_regarray)
y_test = encode_array(y_testdata_regarray)
x_val = encode_array(x_valdata_regarray)
y_val = encode_array(y_valdata_regarray)

x_train=pad_sequences(x_train, maxlen=4381, dtype='int32', padding='post', value=0)
x_test=pad_sequences(x_test, maxlen=4381, dtype='int32', padding='post', value=0)
x_val=pad_sequences(x_val, maxlen=4381, dtype='int32', padding='post', value=0)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

x_train = np.expand_dims(x_train, axis=2)
y_train = np.expand_dims(y_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
y_test = np.expand_dims(y_test, axis=2)
x_val = np.expand_dims(x_val, axis=2)
y_val = np.expand_dims(y_val, axis=2)
print(x_val)
print(y_val)


#########################################

#parameters
filters = 9
kernel_size = 6
stride = 3
dropout_rate = 0.5 #randomly deletes data to improve learning of hyperfeatures
dense_unit = 1 #dimensionality of output space
epochs = 15 #how many times does it stop to assess things such as learning rate

#initiate sequential
rnashape = Sequential()

#1D convolution layers
rnashape.add(Conv1D(10, 6, strides=4, padding='valid', batch_input_shape=(None, 4381, 1)))
rnashape.add(BatchNormalization())
rnashape.add(Activation('relu'))
rnashape.add(Dropout(dropout_rate))
print(rnashape.output_shape)

rnashape.add(Conv1D(30, 6, strides=4, padding='valid'))
rnashape.add(BatchNormalization())
rnashape.add(Activation('relu'))
rnashape.add(Dropout(dropout_rate))
print(rnashape.output_shape)

rnashape.add(Conv1D(50, 7, strides=5, padding='valid'))
rnashape.add(BatchNormalization())
rnashape.add(Activation('relu'))
rnashape.add(Dropout(dropout_rate))
print(rnashape.output_shape)

rnashape.add(Conv1D(75, 4, strides=2, padding='valid'))
rnashape.add(BatchNormalization())
rnashape.add(Activation('relu'))
rnashape.add(Dropout(dropout_rate))
print(rnashape.output_shape)

rnashape.add(Conv1D(100, kernel_size, strides=stride, padding='valid'))
rnashape.add(BatchNormalization())
rnashape.add(Activation('relu'))
rnashape.add(Dropout(dropout_rate))
print(rnashape.output_shape)

#dense
rnashape.add(Dense(dense_unit))
print(rnashape.output_shape)
#rnashape.predict(x_train, batch_size=None, verbose=0)
#optimization layer. Regression problem
rnashape.compile(loss ='mse', optimizer='rmsprop', metrics=['accuracy'])

#model optimization
#hist = rnashape.fit(x_train, y_train, validation_split=0.2)
#print(hist.history)

#training
rnashape.fit(x=x_train, y=y_train, batch_size=None, validation_data=(x_val, y_val), epochs=epochs, shuffle=True, verbose=3)

#testing
score = rnashape.evaluate(x_test, y_test, batch_size=10)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
#plot_model(rnashape, to_file='rnashape.png')
print(score)


#predict
prediction = rnashape.predict(x_train, batch_size=32, verbose=0)
print(prediction)

#trace history
history = rnashape.fit(x=x_train, y=y_train, batch_size=None, shuffle=True, verbose=1, epochs=epochs, validation_data=(x_val, y_val))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('RNAshape accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show(block=False)
plt.savefig('accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNAshape loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show(block=False)
plt.savefig('loss.png')


