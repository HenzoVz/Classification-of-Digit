from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

K.set_image_dim_ordering('tf')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# fix the seed
seed = 7
np.random.seed(seed)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))

classifier.save('classifier_digit.h5')

scores = classifier.evaluate(X_test, y_test, verbose=0)
print("CNN error: %.2f%%" % (100 - scores[1] * 100))