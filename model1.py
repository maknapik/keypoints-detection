from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from matplotlib import pyplot

from data_loader import get_data


X, y = get_data(type="train", flatten=True)

model = Sequential()
model.add(Dense(100, input_dim=2304))
model.add(Activation('relu'))
model.add(Dense(16))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['acc'])
hist = model.fit(X, y, nb_epoch=3000, validation_split=0.2)

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
open('model1_architecture.json', 'w').write(json_string)
model.save_weights('model1_weights.h5')
