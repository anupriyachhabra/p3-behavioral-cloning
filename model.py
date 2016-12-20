
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
import csv
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

    X_train = []
    y_train = []
    with open('driving_log.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            img = misc.imread(row[0])
            #Crop top half image
            height = img.shape[0]
            img = img[height//2 - 25: height-25]
            X_train.append(img)
            steering_angle = float(row[3])
            y_train.append(steering_angle)

            #left image
            left_img = misc.imread(row[1].strip())
            left_img = left_img[height//2 - 25: height-25]
            X_train.append(left_img)
            y_train.append(steering_angle + 0.25)

            #right image
            right_img = misc.imread(row[2].strip())
            right_img = right_img[height//2 - 25: height-25]
            X_train.append(right_img)
            y_train.append(steering_angle - 0.25)

    X_train = preprocess(X_train)


    nb_filters = 32

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

    #model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                           # border_mode='valid'))
    #model.add(Dropout(0.75))

    model.add(Flatten())
    #model.add(Dense(1024))
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
