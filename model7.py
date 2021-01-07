from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.optimizers import SGD
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dropout

from data_loader import get_data
from FlippedImageDataGenerator import FlippedImageDataGenerator

X, y = get_data(type="train", flatten=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=(48, 48, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(16))

start = 0.03
stop = 0.001
nb_epoch = 100
learning_rates = np.linspace(start, stop, nb_epoch)

change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
sgd = SGD(lr=start, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['acc'])
flipgen = FlippedImageDataGenerator(X_train, y_train)
early_stop = EarlyStopping(patience=100)

hist = model.fit_generator(generator=flipgen,
                           samples_per_epoch=X_train.shape[0],
                           nb_epoch=nb_epoch,
                           validation_data=(X_val, y_val),
                           callbacks=[change_lr, early_stop])

pyplot.plot(hist.history['loss'], linewidth=3, label='train')
pyplot.plot(hist.history['val_loss'], linewidth=3, label='valid')
pyplot.grid()
pyplot.legend()
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
pyplot.yscale('log')
pyplot.show()

pyplot.plot(hist.history['acc'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.show()

json_string = model.to_json()
open('model7_architecture.json', 'w').write(json_string)
model.save_weights('model7_weights.h5')
