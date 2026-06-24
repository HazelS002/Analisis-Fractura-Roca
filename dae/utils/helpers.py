import numpy as np
import tensorflow as tf
import cv2

from ..config import resize

def images_to_tensor(images):
    X = [ cv2.resize(img, resize) for img in images ]
    return np.array(X).reshape(-1, resize[0], resize[1], 1)


def tensor_to_images(X):
    return [(img * 255).astype(np.uint8).reshape(resize)\
            for img in X]


def save_model(model, dir):
    model.save(dir)
    return


def load_model(dirname_model):
    return tf.keras.models.load_model(dirname_model)


if __name__ == "__main__": pass