from keras.models import model_from_json
from matplotlib import pyplot

from data_loader import get_data


def plot_sample(x, y, axis, flatten):
    if flatten:
        x = x.reshape(48, 48)
    axis.imshow(x, cmap='gray')
    axis.scatter(y[0::2]*48, y[1::2]*48, marker='x', s=10)


def show_predicts(model_name):
    model = model_from_json(open(model_name + '_architecture.json').read())
    model.load_weights(model_name + '_weights.h5')

    X_test, _ = get_data(type="train", flatten=False)
    y_test = model.predict(X_test)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        axis = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X_test[32 + i], y_test[32 + i], axis, flatten=False)

    pyplot.show()


show_predicts("modelXX")
