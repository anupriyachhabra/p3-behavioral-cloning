
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
import csv
from keras.optimizers import Adam
from scipy import misc
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

def preprocess(images):

    # Normalize
    a = -0.5
    b = 0.5
    min_image = np.min(images)
    max_image = np.max(images)
    images = a + (((images - min_image) * (b - a)) / (max_image - min_image))

    return images


def batch_generator(X_train, Y_train, batch_size = 50):
    train_images = []
    train_steering = []

    while True:
        for i in range(0, len(X_train)):
            img = misc.imread(X_train[i])

            height = img.shape[0]
            width = img.shape[1]

            ''''
            Image has been cropped to remove the unnecessary scenery, so that
            model can focus on learning the important features from the image
            '''
            img = img[height // 2 - 25: height - 25, 50: width-50]

            train_images.append(img)
            train_steering.append(Y_train[i])
            if (i !=0 and i % batch_size == 0) or i == (len(X_train)-1) :
                train_images = preprocess(np.array(train_images))
                yield train_images, np.array(train_steering)
                train_images = []
                train_steering = []


def train_model():

    X_train = []
    y_train = []
    with open('driving_log.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            X_train.append(row[0])
            steering_angle = float(row[3])
            y_train.append(steering_angle)

            #left image 0.25 has been added to steering_angle to get angle equivalent to left camera
            X_train.append(row[1].strip())
            y_train.append(steering_angle + 0.25)

            #right image 0.25 has been subtracted from steering_angle to get angle euivalent to right camera
            X_train.append(row[2].strip())
            y_train.append(steering_angle - 0.25)


    nb_filters = 32

    # convolution kernel size
    kernel_size = (3, 3)
    pool_size =(4, 4)
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(80, 220, 3)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Activation('elu'))
    model.add(Dropout(0.75))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('elu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=Adam())

    # Split for validation data
    X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, y_train,test_size=0.05, random_state=832289)

    # Created checkpoint for checking model's performance at different epochs
    checkpoint_path = "weights.{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')

    model.fit_generator(batch_generator(X_train, Y_train),
                        samples_per_epoch= len(X_train), nb_epoch = 10,
                        verbose=1, validation_data=batch_generator(X_val, Y_val),
                        nb_val_samples=len(X_val), callbacks=[checkpoint], max_q_size=1, pickle_safe=False)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    train_model()
