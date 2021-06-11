from __future__ import print_function

import numpy as np
import os
import pandas as pd
import keras
import nibabel as nib
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

batch_size = 2
bias = 15
num_classes = 90 - bias
epochs = 40

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'new_data')

REFIST_DIR = os.path.join(DATA_DIR, 'IXI-T1')

# data init
try:
    test_files = []
    y = []
    test_labels = []
    train_files = []
    file_path = 'train_X_data.txt'
    with open(file_path, mode='r', encoding='utf-8') as file_obj:
        while True:
            content = file_obj.readline()
            if not content:
                break
            content = content.replace("\n", "")
            train_files.append(content)

    file_path = 'train_Y_data.txt'
    with open(file_path, mode='r', encoding='utf-8') as file_obj:
        while True:
            content = file_obj.readline()
            if not content:
                break
            content = content.replace("\n", "")
            y.append(float(content))

    file_path = 'test_X_data.txt'
    with open(file_path, mode='r', encoding='utf-8') as file_obj:
        while True:
            content = file_obj.readline()
            if not content:
                break
            content = content.replace("\n", "")
            test_files.append(content)

    file_path = 'test_Y_data.txt'
    with open(file_path, mode='r', encoding='utf-8') as file_obj:
        while True:
            content = file_obj.readline()
            if not content:
                break
            content = content.replace("\n", "")
            test_labels.append(float(content))
    print("file init")
except:
    table_path = os.path.join(ROOT_DIR, "new_IXI.csv")
    files_all = [each for each in os.listdir(REFIST_DIR) if not each.startswith('.')]
    df = pd.read_csv(table_path)
    df = df["AGE"]
    y = df.values - bias
    y = y.astype(float)
    train_files, test_files, y, test_labels = train_test_split(files_all, y, test_size=0.1)

    content = ""
    for file in train_files:
        content = content + str(file) + '\n'
    file_path = 'train_X_data.txt'
    with open(file_path, mode='w', encoding='utf-8') as file_obj:
        file_obj.write(content)

    content = ""
    for file in test_files:
        content = content + str(file) + '\n'
    file_path = 'test_X_data.txt'
    with open(file_path, mode='w', encoding='utf-8') as file_obj:
        file_obj.write(content)

    content = ""
    for file in y:
        content = content + str(file) + '\n'
    file_path = 'train_Y_data.txt'
    with open(file_path, mode='w', encoding='utf-8') as file_obj:
        file_obj.write(content)

    content = ""
    for file in test_labels:
        content = content + str(file) + '\n'
    file_path = 'test_Y_data.txt'
    with open(file_path, mode='w', encoding='utf-8') as file_obj:
        file_obj.write(content)

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
    model.add(Conv3D(8, (3, 3, 3), activation='relu',
                     padding='same', name='conv4'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', name='full_connect'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu', name='final'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.summary()

    # train model
    train_x = []
    for f in train_files:
        img = nib.load(os.path.join(REFIST_DIR, f))
        img_data = img.get_fdata()
        img_data = np.asarray(img_data)
        train_x.append(img_data)
    print("database_added")
    x_train = np.asarray(train_x)
    x_train = np.expand_dims(x_train, 4)
    y_train = np.asarray(y)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, verbose=1)
    model.save('my_model.h5')
    print("training finished")
print("model init")

# evaluate
test_x, test_y = [], test_labels
for f in test_files:
    img = nib.load(os.path.join(REFIST_DIR, f))
    img_data = img.get_fdata()
    img_data = np.asarray(img_data)
    test_x.append(img_data)

test_x = np.asarray(test_x)
x_test = np.expand_dims(test_x, 4)
y_test = np.asarray(test_y)
mae = model.evaluate(x_test, y_test)
print('\n\n MAE is: ', mae)

pre = model.predict(x_test)
print(pre)
print(y_test)