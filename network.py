from __future__ import print_function

import numpy as np
import os
import pandas as pd
import keras
import nibabel as nib
from keras import callbacks
from keras.layers.core import Flatten, Dropout
from sklearn.metrics import mean_absolute_error
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import Input, Dense, AlphaDropout
from sklearn.model_selection import train_test_split

batch_size = 2
epochs = 10
age_range = 10
bias = int(20 / age_range)
num_classes = int(90 / age_range) - bias

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'new_data')
REFIST_DIR = os.path.join(DATA_DIR, 'IXI-T1')


def loading_file(file_path):
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


# data init
try:
    init_train_x = []
    init_train_y = []
    init_test_x = []
    init_test_y = []
    file_path = 'train_X_data.txt'
    init_train_x = loading_file(file_path)
    file_path = 'train_Y_data.txt'
    init_train_y = loading_file(file_path)
    file_path = 'test_X_data.txt'
    init_test_x = loading_file(file_path)
    file_path = 'test_Y_data.txt'
    init_test_y = loading_file(file_path)
    print("file init")
except:
    table_path = os.path.join(ROOT_DIR, "new_IXI.csv")
    files_all = [each for each in os.listdir(REFIST_DIR) if not each.startswith('.')]
    df = pd.read_csv(table_path)
    df = df["AGE"]
    labels = np.round(df.values) / age_range - bias
    labels = labels.astype(int)
    init_train_x, init_test_x, init_train_y, init_test_y = train_test_split(files_all,
                                                                            labels, test_size=0.2)
    file_path = 'train_X_data.txt'
    add_file(file_path, init_train_x)
    file_path = 'test_X_data.txt'
    add_file(file_path, init_test_x)
    file_path = 'train_Y_data.txt'
    add_file(file_path, init_train_y)
    file_path = 'test_Y_data.txt'
    add_file(file_path, init_test_y)
    print("code init")
print("database init")

# construct model
try:
    model = keras.models.load_model('my_model.h5')
except:
    dimx, dimy, channels = 91, 109, 91
    model = keras.Sequential()
    model.add(Input(shape=(dimx, dimy, channels, 1), name="input"))
    model.add(Conv3D(2, (3, 3, 3), activation='relu',
                     padding='same', name='conv1'))
    model.add(Conv3D(4, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    model.add(Conv3D(8, (3, 3, 3), activation='relu',
                     padding='same', name='conv3'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', name='full_connect'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics="accuracy")
    model.summary()

if epochs != 0:
    # train model
    train_x = []
    train_files, val_files, train_labels, val_labels = train_test_split(init_train_x,
                                                                        init_train_y, test_size=0.1)
    x_train = load_image(train_files)
    y_train = to_categorical(train_labels, num_classes)

    x_val = load_image(val_files)
    y_val = to_categorical(val_labels, num_classes)
    print("database_added")

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, verbose=1, validation_data=(x_val, y_val))
    model.save('my_model.h5')
    print("training finished")
    print("model init")

# evaluate
x_test = load_image(init_test_x)
y_test = to_categorical(init_test_y, num_classes)
acc = model.evaluate(x_test, y_test)
print('\n\n accuracy is: ', acc[1])

pred = model.predict([x_test])
pred = [i.argmax() for i in pred]
mae = mean_absolute_error(init_test_y, pred)
print('\n\n MAE is: ', mae)
