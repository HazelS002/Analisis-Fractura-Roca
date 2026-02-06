import numpy as np
import cv2 as cv2
from sklearn.decomposition import NMF

def apply_nmf(images: list[np.ndarray], **nmf_params)\
    -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
    
    # Convertir la lista de imágenes a una matriz 2D
    n_samples, images_shape = len(images), images[0].shape
    data_matrix = np.array([img.flatten() for img in images])

    # Aplicar NMF
    nmf_model = NMF(**nmf_params)
    print(f"Fitting NMF model, {nmf_params['n_components']} components...")
    W = nmf_model.fit_transform(data_matrix)
    H = nmf_model.components_

    # Reconstruir las imágenes a partir de las componentes
    reconstructed_images = [ (np.dot(W[i,:], H).reshape(images_shape))\
                            .astype(np.uint8) for i in range(n_samples)]

    return reconstructed_images, W, H, nmf_model.reconstruction_err_


if __name__ == "__main__":
    from src import SAMPLE_DATA_DIR, NMF_PARAMS
    from src.load_images import read_images
    from src.visualitation import show_images, show_nmf_components,\
        plot_reconstruction_error
    from src.transform import clean_images
    from src.utils import get_stage_images

    
    # leer y limpiar imagenes
    images, names = read_images(SAMPLE_DATA_DIR + "aligned-images/")
    images = get_stage_images(clean_images(images), "cleaned")
    
    # promediar imagenes
    show_images([np.mean(images, axis=0).astype(np.uint8)], ["Average Image"],
                suptitle="Average of Aligned and Cleaned Images")

    # Evaluar error de reconstrucción para diferentes números de componentes
    errors = [ apply_nmf(images, n_components=k, **NMF_PARAMS)[3]\
              for k in range(1, 12) ]
    plot_reconstruction_error(errors, list(range(1, 12)))

    # aplicar NMF
    for k in [2, 10]:
        print(f"\nNMF con {k} componentes...")
        nmf_images, W, H, _ = apply_nmf(images, n_components=k, **NMF_PARAMS)
        show_nmf_components(H, W, images[0].shape)
        show_images(nmf_images, names,
                    suptitle=f"NMF Reconstruction with {k} Components")