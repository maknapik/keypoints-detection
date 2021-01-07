import os
import re
from PIL import Image
import numpy as np

data_directory = "./Keypoints/"
object_names = ["drill", "duck", "extension", "frog", "multimeter", "pig"]


def get_data(type="train", flatten=False):
    images = list()
    labels = list()
    for object_name in object_names:
        files = [f for f in os.listdir(data_directory + object_name + "/" + type) if re.match(r'^[0-9]{5}-mask\.png$', f)]
        for file in sorted(files):
            image = Image.open(data_directory + object_name + "/" + type + "/" + file)
            image = image.resize((48, 48), Image.ANTIALIAS)
            if flatten:
                image = image.convert('L')
                image = np.asarray(image).reshape(2304)
            else:
                image = np.asarray(image)
            images.append(image)
        keypoints = np.load(data_directory + object_name + "/" + object_name + "_keypoints_" + type + ".npy")
        tolist = keypoints.reshape(keypoints.shape[0], 16).tolist()
        labels.append(tolist)

    flatten_labels = [item for sublist in labels for item in sublist]

    return np.array(images), np.array(flatten_labels)
