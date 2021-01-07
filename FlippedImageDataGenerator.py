import numpy as np
import keras


class FlippedImageDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, dim=(48, 48, 3), shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = y
        self.list_IDs = X
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 16))

        for index, batch_index in zip(indexes, range(0, self.batch_size)):
            X[batch_index] = self.list_IDs[index, :, ::-1, :]
            y[batch_index] = self.labels[index]

        return X, y
