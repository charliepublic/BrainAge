from __future__ import print_function

import numpy as np
import nibabel as nib
import os
import pandas as pd
import keras

from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers import Input, Lambda, Embedding, Bidirectional, LSTM, Dense
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print (keras.backend.image_data_format())

batch_size = 10
num_classes = 2
epochs = 20
learning_rate = 0.01
decay = 1e-6
validation_split = 0.2


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,'normalisation')

REFIST_DIR = os.path.join(DATA_DIR,'IXI-T1')
# DEMO_DIR = os.path.join(ROOT_DIR,'demographic')

table_path = os.path.join(ROOT_DIR,"IXI.xls")

files_all = [each for each in os.listdir(REFIST_DIR) if not each.startswith('.')]
df = pd.read_excel(table_path)
print(df)



y = df.values
y = y.astype(int)

train_files, test_files, y, test_labels = train_test_split(files_all, y, test_size=0.1)

# construct model
dimx,dimy,channels = 182,218,144
inpx = Input(shape=(dimx,dimy,channels,1),name='inpx')
x = Convolution3D(2, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv1')(inpx)
x = Convolution3D(4, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv2')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       border_mode='valid', name='pool2')(x)
x = Convolution3D(8, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv3')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       border_mode='valid', name='pool3')(x)
hx = Flatten()(x)
score = Dense(2, activation='softmax', name='fc8')(hx)
model = Model(inputs=inpx, outputs=score)
opt = keras.optimizers.rmsprop(lr=learning_rate, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


# train model
train_x = []
for f in train_files:
    img = nib.load(os.path.join(REFIST_DIR, f))
    img_data = img.get_data()
    img_data = np.asarray(img_data)
    img_data = img_data[:,:,0:144]
    train_x.append(img_data)
x_train = np.asarray(train_x)
print('\n iteration number :', 1, '\n')
x_train = np.expand_dims(x_train, 4)
print('\n', x_train.shape)
y_train = y
y_train = keras.utils.to_categorical(y_train, num_classes)

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, verbose=2,validation_split=validation_split)

# evaluate
test_x,test_y = [], test_labels
for i,f in enumerate(test_files):
    img = nib.load(os.path.join(REFIST_DIR, f))
    img_data = img.get_data()
    img_data = np.asarray(img_data)
    img_data = img_data[:, :, 0:144]
    test_x.append(img_data)
test_x = np.asarray(test_x)
test_x = np.expand_dims(test_x,4)
test_y = keras.utils.to_categorical(test_y, num_classes)
pred = model.predict([test_x])

pred = [i.argmax() for i in pred]



mae = mean_absolute_error(test_labels,pred)
print('\n\n MAE is: ', mae)
mse = mean_squared_error(test_labels,pred)
print('\n\n MSE is: ', mse)

