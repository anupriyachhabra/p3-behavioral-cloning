
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
from keras.optimizers import Adam
from scipy import misc
from sklearn.model_selection import train_test_split

import pandas as pd


training_data = pd.read_csv("driving_log.csv", header=None)
X_train = []
images = training_data[0]
for i in range(0, len(images)):
    X_train.append(misc.imread(images[i]))
y_train = training_data[3]



a = -0.5
b = 0.5
min_image = np.min(X_train)
max_image = np.max(X_train)
X_train =  a + (((X_train - min_image)*(b - a))/( max_image - min_image))


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

model.add(Flatten())
model.add(Dense(256))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])


X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, y_train,test_size=0.25, random_state=832289)

history = model.fit(X_train, Y_train,
                    batch_size=50, nb_epoch = 10,
                    verbose=1, validation_data=(X_val, Y_val))
print ("Validation Accuracy ", history.history['val_acc'][0])
