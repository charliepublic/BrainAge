from __future__ import print_function

import os

import keras
import keras.backend as K
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks
from keras.engine.input_layer import InputLayer
from keras.layers import Dense
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

batch_size = 8
epochs = 50
num_classes = 2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
flag = "GM"

if flag == "GM":
    DATA_DIR = os.path.join(ROOT_DIR, 'segment_GM')
elif flag == "CFS":
    DATA_DIR = os.path.join(ROOT_DIR, 'segment_cfs')
elif flag == "WM":
    DATA_DIR = os.path.join(ROOT_DIR, 'segment_WM')
elif flag == "CFS+GM":
    DATA_DIR = os.path.join(ROOT_DIR, 'segment_GM+CFS')
elif flag == "CFS+WM":
    DATA_DIR = os.path.join(ROOT_DIR, 'segment_WM+CFS')
elif flag == "WM+GM":
    DATA_DIR = os.path.join(ROOT_DIR, 'segment_WM+GM')
else:
    # the whole brain
    flag = "Whole brain"
    DATA_DIR = os.path.join(ROOT_DIR, 'new_data')

REFIST_DIR = os.path.join(DATA_DIR, 'IXI-T1')

if flag == "whole brain":
    logfile_path = "logfile/"
else:
    logfile_path = "Segment_logfile/"


def loading_file(file_path):
    file_path = logfile_path + file_path
    return_list = []
    with open(file_path, mode='r', encoding='utf-8') as file_obj:
        while True:
            content = file_obj.readline()
            if not content:
                break
            content = content.replace("\n", "")
            return_list.append(content)
    return return_list


def add_file(file_path, target_list):
    content = ""
    file_path = logfile_path + file_path
    for file in target_list:
        content = content + str(file) + '\n'
    with open(file_path, mode='w', encoding='utf-8') as file_obj:
        file_obj.write(content)


def load_image(file_list):
    x = []
    for f in file_list:
        img = nib.load(os.path.join(REFIST_DIR, f))
        img_data = img.get_fdata()
        img_data = np.asarray(img_data)
        x.append(img_data)
    x = np.asarray(x)
    x = np.expand_dims(x, 4)
    return x


# binary focal loss to solve uneven data
def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


# data init
try:
    file_path = 'train_X_data.txt'
    train_files = loading_file(file_path)
    file_path = 'train_Y_data.txt'
    train_labels = loading_file(file_path)

    file_path = 'test_X_data.txt'
    init_test_x = loading_file(file_path)
    file_path = 'test_Y_data.txt'
    init_test_y = loading_file(file_path)

    file_path = 'val_x_data.txt'
    val_files = loading_file(file_path)
    file_path = 'val_Y_data.txt'
    val_labels = loading_file(file_path)
    print("file init")
except:
    if flag == "whole brain":
        table_path = os.path.join(ROOT_DIR, "new_IXI.csv")
    else:
        table_path = os.path.join(ROOT_DIR, "new_IXI_segment.csv")

    files_all = [each for each in os.listdir(REFIST_DIR) if not each.startswith('.')]
    df = pd.read_csv(table_path)
    df = df["AGE"]
    labels = np.round(df.values) / 50
    labels = labels.astype(int)
    init_train_x, init_test_x, init_train_y, init_test_y = train_test_split(files_all,
                                                                            labels, test_size=0.1)

    train_files, val_files, train_labels, val_labels = train_test_split(init_train_x,
                                                                        init_train_y, test_size=0.1)

    file_path = 'train_X_data.txt'
    add_file(file_path, train_files)
    file_path = 'train_Y_data.txt'
    add_file(file_path, train_labels)

    file_path = 'test_X_data.txt'
    add_file(file_path, init_test_x)
    file_path = 'test_Y_data.txt'
    add_file(file_path, init_test_y)

    file_path = 'val_x_data.txt'
    add_file(file_path, val_files)
    file_path = 'val_Y_data.txt'
    add_file(file_path, val_labels)

    print("code init")
print("database init")

# construct model
# #Possible recommendations: It is not recommended to use 3D images. It should be converted to 2d to
# increase the number of samples, For example each 3D image should be taken some part of the slice for training. This
# may need  more time  to reconstruct the network and adjust the hyper_parameters
model_name = flag + "_new_model_2_best.h5"
try:
    model = keras.models.load_model(model_name)
except:
    print("code init start")
    dimx, dimy, channels = 91, 109, 91
    model = keras.Sequential()
    model.add(InputLayer(input_shape=(dimx, dimy, channels, 1), name="input"))
    model.add(Conv3D(2, (3, 3, 3), activation='relu',
                     padding='same', name='conv1'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool2'))
    model.add(Conv3D(4, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(Conv3D(8, (3, 3, 3), activation='relu',
                     padding='same', name='conv3'))
    model.add(Conv3D(4, (3, 3, 3), activation='relu',
                     padding='same', name='conv4'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool3'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, use_bias=True, activation='softmax', name='full_connect'))
    sgd = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics="accuracy")
    model.summary()
    print("code init finished")

if epochs != 0:
    # train model
    train_x = []
    print("loading data")
    x_train = load_image(train_files)
    y_train = to_categorical(train_labels, num_classes)
    print("train_database_added")
    x_val = load_image(val_files)
    y_val = to_categorical(val_labels, num_classes)
    print("val_database_added")

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=2, verbose=0, mode='auto')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping])

    model.save(model_name)

    del x_train
    print("training finished")
    print("model init")

# evaluate
x_test = load_image(init_test_x)
y_test = to_categorical(init_test_y, num_classes)
acc = model.evaluate(x_test, y_test, batch_size=2)
print('\n\n accuracy is: ', acc[1])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print(loss)
print(val_acc)
epochs = range(1, len(acc) + 1)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epochs, acc, 'r', label='Trainning acc')
plt.plot(epochs, val_acc, 'b', label='Vaildation acc')
plt.legend()
plt.show()

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(epochs, loss, 'r', label='Trainning loss')
plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
plt.legend()
plt.show()
