import numpy as np
import cv2 as cv2
from sklearn.decomposition import NMF

def apply_nmf(images: list[np.ndarray], **nmf_params)\
    -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    
    # Convertir la lista de imágenes a una matriz 2D
    n_samples, images_shape = len(images), images[0].shape
    data_matrix = np.array([img.flatten() for img in images])

    # Aplicar NMF
    nmf_model = NMF(**nmf_params)
    W = nmf_model.fit_transform(data_matrix)
    H = nmf_model.components_

    # Reconstruir las imágenes a partir de las componentes
    reconstructed_images = [ (np.dot(W[i,:], H).reshape(images_shape))\
                            .astype(np.uint8) for i in range(n_samples)]

    return reconstructed_images, W, H


if __name__ == "__main__":
    from src import SAMPLE_DATA_DIR
    from src.load_images import read_images
    from src.visualitation import show_images, show_nmf_components
    from src.transform import clean_images
    from src.utils import get_stage_images

    # leer y limpiar imagenes
    images, names = read_images(SAMPLE_DATA_DIR + "aligned-images/")
    images = get_stage_images(clean_images(images), "cleaned")

    # # ver imagenes
    # show_images(images, names)

    # # promediar imagenes
    # from matplotlib import pyplot as plt
    #
    # result = np.mean(images, axis=0).astype(np.uint8)
    # plt.imshow(result, cmap="gray")
    # plt.show()

    # aplicar NMF
    nmf_images, W, H = apply_nmf(images, n_components=4, max_iter=500,
                                 random_state=42)
    show_images(nmf_images, names)

    show_nmf_components(H, W, images[0].shape)


    