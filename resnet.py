from __future__ import print_function

import numpy as np
import os
import pandas as pd
import keras
import nibabel as nib
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from resnet3d import Resnet3DBuilder


batch_size = 1
epochs = 0
age_range = 1
bias = int(20 / age_range)
num_classes = 2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'new_data')
REFIST_DIR = os.path.join(DATA_DIR, 'IXI-T1')


def loading_file(file_path):
    file_path = "logfile/" + file_path
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
    file_path = "logfile/"+file_path
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
    table_path = os.path.join(ROOT_DIR, "new_IXI.csv")
    files_all = [each for each in os.listdir(REFIST_DIR) if not each.startswith('.')]
    df = pd.read_csv(table_path)
    df = df["AGE"]
    labels = np.round(df.values)/50
    labels = labels.astype(int)
    init_train_x, init_test_x, init_train_y, init_test_y = train_test_split(files_all,
                                                                            labels, test_size=0.1)

    train_files, val_files, train_labels, val_labels = train_test_split(init_train_x,
                                                                        init_train_y, test_size=0.1)

    file_path = 'logfile/train_X_data.txt'
    add_file(file_path, train_files)
    file_path = 'logfile/train_Y_data.txt'
    add_file(file_path, train_labels)

    file_path = 'logfile/test_X_data.txt'
    add_file(file_path, init_test_x)
    file_path = 'logfile/test_Y_data.txt'
    add_file(file_path, init_test_y)

    file_path = 'logfile/val_x_data.txt'
    add_file(file_path, val_files)
    file_path = 'logfile/val_Y_data.txt'
    add_file(file_path, val_labels)

    print("code init")
print("database init")

# construct model
model_name = 'res_model_34_2.h5'
try:
    model = keras.models.load_model(model_name)
except:
    dimx, dimy, channels = 91, 109, 91
    model = Resnet3DBuilder.build_resnet_10((91, 109, 91, 1), 2)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

if epochs != 0:
    # train model
    train_x = []

    x_train = load_image(train_files)
    y_train = to_categorical(train_labels, num_classes)

    x_val = load_image(val_files)
    y_val = to_categorical(val_labels, num_classes)
    print("database_added")

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                   patience=3, verbose=0, mode='auto')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, verbose=1, validation_data=(x_val, y_val),callbacks = [early_stopping])
    model.save(model_name)
    del x_train
    print("training finished")
    print("model init")

# evaluate
x_test = load_image(init_test_x)
# y_test = np.asarray(init_test_y)
# y_test = y_test.astype(float)
y_test = to_categorical(init_test_y, num_classes)
acc = model.evaluate(x_test, y_test,batch_size=2)
print('\n\n accuracy is: ', acc[1])

pred = model.predict([x_test],batch_size=2)
pred = [i.argmax() for i in pred]
init_test_y = np.asarray(init_test_y)
init_test_y = init_test_y.astype(int)
mae = mean_absolute_error(init_test_y, pred)
print('\n\n MAE is: ', mae)
