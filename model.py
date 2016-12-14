
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
import cv2
from keras.optimizers import Adam
from scipy import misc
from sklearn.model_selection import train_test_split

import pandas as pd

def preprocess(images):

    # Normalize
    a = -0.5
    b = 0.5
    min_image = np.min(images)
    max_image = np.max(images)
    images = a + (((images - min_image) * (b - a)) / (max_image - min_image))

    return images

def train_model():

    training_data = pd.read_csv("driving_log.csv", header=None)
    X_train = []
    images = training_data[0]

    for i in range(0, len(images)):
        img = misc.imread(images[i])
         #Crop top half image
        img = img[img.shape[0]//2:]
        X_train.append(img)
    y_train = training_data[3]

    X_train = preprocess(X_train)
    nb_filters = 256

    # convolution kernel size
    kernel_size = (3, 3)
    pool_size =(4, 4)
    input_shape = X_train.shape[1:]
    print(input_shape)
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.75))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])


    X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, y_train,test_size=0.05, random_state=832289)

    history = model.fit(X_train, Y_train,
                        batch_size=50, nb_epoch = 3,
                        verbose=1, validation_data=(X_val, Y_val))
    print ("Validation Accuracy ", history.history['val_acc'][0])

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    train_model()
