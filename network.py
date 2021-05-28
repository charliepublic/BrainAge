from __future__ import print_function

import numpy as np
import os
import pandas as pd
import keras
from tensorflow.keras.utils import to_categorical
import nibabel as nib
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import Input, Dense
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

batch_size = 2
num_classes = 100
epochs = 10
rate = 70
train_number = 490
learning_rate = 0.01
decay = 1e-6
validation_split = 0.2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'normalisation')

REFIST_DIR = os.path.join(DATA_DIR, 'IXI-T1')
# DEMO_DIR = os.path.join(ROOT_DIR,'demographic')
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
            y.append(content)

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
            test_labels.append(content)
    print("file init")
except:
    table_path = os.path.join(ROOT_DIR, "new_IXI.csv")
    files_all = [each for each in os.listdir(REFIST_DIR) if not each.startswith('.')]
    df = pd.read_csv(table_path)
    df = df["AGE"]
    y = df.values
    y = y.astype(int)
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
    dimx, dimy, channels = 182, 218, 144
    model = keras.Sequential()
    model.add(Input(shape=(dimx, dimy, channels, 1), name='inpx'))
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
    model.add(Conv3D(4, (3, 3, 3), activation='relu',
                     padding='same', name='conv5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', name='fc8'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
print("model init")
# train model
train_x = []
try:
    file_path = 'data.txt'
    with open(file_path, mode='r', encoding='utf-8') as file_obj:
        i = int(file_obj.readline())
except:
    i = 0

train_files = train_files[i:]
for f in train_files:
    if i == train_number:
        break
    img = nib.load(os.path.join(REFIST_DIR, f))
    img_data = img.get_fdata()
    i = i + 1
    img_data = np.asarray(img_data)
    img_data = img_data[:, :, 0:144]
    train_x.append(img_data)

    if i % rate == 0 or i == train_number:
        times = int(i / rate)
        x_train = np.asarray(train_x)
        x_train = np.expand_dims(x_train, 4)
        if i != train_number:
            y_train = y[(times - 1) * rate:times * rate]
        else:
            y_train = y[times * rate:]

        y_train = to_categorical(y_train, num_classes)

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs, verbose=2)
        print("______________________________________")
        model.save('my_model.h5')
        print(i)
        file_path = 'data.txt'
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            file_obj.write(str(i))
        train_x = []

print("______________________________________")
print("training finished")

# evaluate
test_x, test_y = [], test_labels
i = 0
rate = 5
for f in test_files:
    img = nib.load(os.path.join(REFIST_DIR, f))
    img_data = img.get_fdata()
    img_data = np.asarray(img_data)
    img_data = img_data[:, :, 0:144]
    test_x.append(img_data)
    i = i + 1
    if i % rate == 0:
        times = int(i / rate)
        test_x = np.asarray(test_x)
        test_x = np.expand_dims(test_x, 4)
        if i == train_number:
            y_test = y[times * rate:]
        else:
            y_test = test_y[(times - 1) * rate:times * rate]

        pred = model.predict([test_x])

        pred = [i.argmax() for i in pred]
        print(pred)
        print(y_test)
        mae = mean_absolute_error(y_test, pred)
        print('\n\n MAE is: ', mae)
        test_x = []