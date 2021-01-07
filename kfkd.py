from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.optimizers import SGD
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dropout

from data_loader import get_data

X, y = get_data(type="train", flatten=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model_top = Sequential()
model_top.add(Flatten(input_shape=X_train.shape[1:]))
model_top.add(Dense(1000))
model_top.add(Activation('relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(1000))
model_top.add(Activation('relu'))
model_top.add(Dense(16))

start = 0.01
stop = 0.001
nb_epoch = 300

sgd = SGD(lr=start, momentum=0.9, nesterov=True)
model_top.compile(loss='mean_squared_error', optimizer=sgd, metrics=['acc'])

early_stop = EarlyStopping(patience=100)
learning_rates = np.linspace(start, stop, nb_epoch)
change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))

hist = model_top.fit(X_train, y_train,
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

json_string = model_top.to_json()
open('kfkd_architecture.json', 'w').write(json_string)
model_top.save_weights('kfkd_weights.h5')
