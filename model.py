
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ELU
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
import cv2
from keras.optimizers import Adam
from scipy import misc
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

import pandas as pd

def preprocess(images):

    # Normalize
    a = -0.5
    b = 0.5
    min_image = np.min(images)
    max_image = np.max(images)
    images = a + (((images - min_image) * (b - a)) / (max_image - min_image))

    return images


def batch_generator(X_train, Y_train , input_shape, batch_size = 64):
    num_rows = len(X_train)

    train_images = np.zeros((batch_size, input_shape[0], input_shape[1], 3))
    train_steering = np.zeros(batch_size)
    ctr = None
    while 1:
        for j in range(batch_size):
            if ctr is None or ctr >= num_rows:
                ctr = 0     # Initialize counter or reset counter if over bounds
            train_images[j] = X_train[ctr]
            train_steering[j] = Y_train[ctr]
            ctr += 1
        train_images = preprocess(train_images)
        yield train_images, train_steering


def train_model():

    training_data = pd.read_csv("driving_log.csv", header=None)
    X_train = []
    images = training_data[0]

    for i in range(0, len(images)):
        img = misc.imread(images[i])
         #Crop top half image
        img = img[img.shape[0]//3:]
        X_train.append(img)
    y_train = training_data[3]

    input_shape = X_train[0].shape
    print(input_shape)

    # Nvidea model
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample= (2, 2), name='conv1_1', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample= (2, 2), name='conv2_1'))
    model.add(Dropout(0.75))
    model.add(Convolution2D(48, 5, 5, subsample= (2, 2), name='conv3_1'))
    model.add(Activation('elu'))
    model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='conv4_1'))
    model.add(Activation('elu'))
    model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='conv4_2'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164, name = "dense_0"))
    model.add(Activation('elu'))
    model.add(Dense(100, name = "dense_1"))
    model.add(Dense(50, name = "dense_2"))
    model.add(Dense(10, name = "dense_3"))
    model.add(Dense(1, name = "dense_4"))

    model.compile(loss='mean_squared_error', optimizer=Adam())


    X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, y_train,test_size=0.1, random_state=832289)

    checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    model.fit_generator(batch_generator(X_train, np.array(Y_train), input_shape), samples_per_epoch = len(X_train),
                        nb_epoch = 3, verbose=1, callbacks=[checkpoint],
                        validation_data= batch_generator(X_val, np.array(Y_val), input_shape), nb_val_samples=len(X_val))


    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    train_model()
