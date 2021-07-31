import os
import random

import keras
import keras.backend as K
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.engine.input_layer import InputLayer
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dropout, Dense
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD


# use binary focal loss to solve the uneven dataset samples
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


# repeat the amount of the dataset to the expected amount
def fit_to_number(original_list, number):
    fitted_list = original_list
    original_len = len(original_list)
    if original_list > number:
        print(original_list)
        print("error" + str(number))
        exit()
    times = int(number / original_list)
    rest = number % original_list
    i = 0
    for i in range(times - 1):
        fitted_list = fitted_list + original_list
    fitted_list = fitted_list + original_list[0:rest]
    if len(fitted_list) != number:
        exit()
    return fitted_list


# split the dataset by 6:1:1
def get_rearrange(original_list):
    length = len(original_list)
    flag1 = int(length / 4 * 3)
    flag2 = int(length / 8 * 7)
    train = original_list[0:flag1]
    test = original_list[flag1:flag2]
    val = original_list[flag2: length]

    train = fit_to_number(train, 90)
    val = fit_to_number(val, 15)
    test = fit_to_number(test, 15)
    return train, val, test


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


epochs = 10
batch_size = 8
dic_list = ["segment_cfs", "segment_GM", "segment_WM", "segment_GM+CFS",
            "segment_WM+CFS", "segment_WM+GM"]
filePath = 'reload_data/'
name_list = os.listdir(filePath)
file_path = "accuracy_log.txt"
for gap in range(1, 7):
    for item in dic_list:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(ROOT_DIR, item)
        REFIST_DIR = os.path.join(DATA_DIR, 'IXI-T1')
        content = "gap is " + str(gap) + "   " + str(item) + "\n"
        with open(file_path, mode='a+', encoding='utf-8') as file_obj:
            file_obj.write(content)

        for start_number in range(2, 9 - gap):
            end_number = start_number + gap
            # init train data, val data, test data
            target_filePath = filePath + str(start_number) + "/"
            fileA = os.listdir(target_filePath)
            trainA, valA, testA = get_rearrange(fileA)
            target_filePath = filePath + str(end_number) + "/"
            fileB = os.listdir(target_filePath)
            trainB, valB, testB = get_rearrange(fileB)

            train_files = trainA + trainB
            train_labels = [0] * len(trainA) + [1] * len(trainB)
            cc = list(zip(train_files, train_labels))
            random.shuffle(cc)
            train_files[:], train_labels[:] = zip(*cc)

            val_files = valA + valB
            val_labels = [0] * len(valA) + [1] * len(valB)

            init_test_x = testA + testB
            init_test_y = [0] * len(testA) + [1] * len(testB)

            print("code init start")
            # construct the model
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
            model.add(Dense(2, use_bias=True, kernel_regularizer="l2", activation='softmax', name='full_connect'))
            sgd = SGD()
            model.compile(loss='categorical_crossentropy', optimizer="adam", metrics="accuracy")
            model.summary()
            print("code init finished")

            if epochs != 0:
                # train model
                train_x = []
                print("loading data")

                x_train = load_image(train_files)
                y_train = to_categorical(train_labels, 2)
                print("train_database_added")
                x_val = load_image(val_files)
                y_val = to_categorical(val_labels, 2)
                print("val_database_added")

                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=3, verbose=0, mode='auto')
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping])

                del x_train
                print("training finished")
                print("model init")

            # evaluate
            x_test = load_image(init_test_x)
            y_test = to_categorical(init_test_y, 2)
            acc = model.evaluate(x_test, y_test, batch_size=2)
            print('\n\n accuracy is: ', acc[1])

            file_path = "accuracy_log.txt"
            content = "start number is " + str(start_number) + " end number is " + str(
                start_number + gap) + " accuracy is: " + str(acc[1]) + "\n"
            with open(file_path, mode='a+', encoding='utf-8') as file_obj:
                file_obj.write(content)

        content = "____________________________\n"
        with open(file_path, mode='a+', encoding='utf-8') as file_obj:
            file_obj.write(content)
