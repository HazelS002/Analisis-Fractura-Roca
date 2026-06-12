import numpy as np
import cv2 as cv2
from sklearn.decomposition import NMF
from .config import nmf_kwargs


def apply_nmf(images: list[np.ndarray], n_components:int):
    # Convertir la lista de imágenes a una matriz 2D
    n_samples, images_shape = len(images), images[0].shape
    data_matrix = np.array([img.flatten() for img in images])

    # Aplicar NMF
    nmf_model = NMF(n_components=n_components, **nmf_kwargs)
    W = nmf_model.fit_transform(data_matrix)
    H = nmf_model.components_

    # Reconstruir las imágenes a partir de las componentes
    reconstructed_images = [ (np.dot(W[i,:], H).reshape(images_shape))\
                            .astype(np.uint8) for i in range(n_samples)]

    return reconstructed_images, W, nmf_model

if __name__ == "__main__": pass