import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .utils.helpers import tensor_to_images, images_to_tensor
from .config import *

def set_dae():
    # Capa de entrada
    input_img = layers.Input(shape=(resize[0], resize[1], 1))
    
    # --- CLAVE DEL DENOISING: Añadir ruido gaussiano solo a la entrada ---
    # Desviación estándar de 0.1 (10% de ruido) obliga al modelo a esforzarse por limpiar
    # noise_layer = layers.GaussianNoise(0.1)(input_img)
    
    # ENCODER (Comprimir la imagen)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # (Bottleneck: La imagen de 256x256 se ha comprimido a 32x32 con 8 canales.
    # Aquí NO cabe el ruido aleatorio, solo cabe la estructura del arco).
    
    # DECODER (Reconstruir la imagen limpia)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Crear el modelo
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(**c_kwargs)
    return autoencoder


def train_dae(model, images):
    X = images_to_tensor(images)
    history = model.fit(X, X, **mf_kwargs)
    return history

def predict(model, images):
    images_size = images[0].shape
    X = images_to_tensor(images)
    Y = model.predict(X)
    return tensor_to_images(Y)

if __name__ == "__main__": pass