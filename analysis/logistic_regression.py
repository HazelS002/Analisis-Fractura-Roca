import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import lr_kwargs, rimages_weight, fimages_weight



def _create_poisson_images(shape: tuple[int, int], n_images: int,
                          lam: float, random_state=42) -> list[np.ndarray]:
    """
    Genera imágenes aleatorias cuyos píxeles siguen una Poisson(lam).

    Recibe:
        shape        : tuple[int, int]. Dimensiones (alto, ancho).
        n_images     : int. Número de imágenes a generar.
        lam          : float. Parámetro lambda de la Poisson.
        random_state : int. Semilla.

    Devuelve:
        list[np.ndarray] con imágenes uint8 en [0, 255].
    """
    rng = np.random.default_rng(random_state)

    return [
        rng.poisson(lam=lam, size=shape).clip(0, 255).astype(np.uint8)
        for _ in range(n_images)
    ]


def create_data(images: list[np.ndarray], lam:float, shape, images_proportion:\
                float = 1.0) -> tuple[list[np.ndarray], list[int]]:
    
    # crear imagenes aleatorias
    n_fakeimages = int(np.round(images_proportion*len(images)))
    fake_images = _create_poisson_images(shape, n_fakeimages, lam)

    # concatenar y etiquetar imagenes
    all_images = images.copy() + fake_images
    labels = [1]*len(images) + [0]*n_fakeimages

    # permutar imagenes
    perm = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in perm]
    labels = [labels[i] for i in perm]

    return np.array(all_images), np.array(labels)


def _sample_weight(labels: np.ndarray):
    return None if rimages_weight is None or fimages_weight is None else\
        np.where(labels == 1, rimages_weight, fimages_weight)


def get_mask(lr: LogisticRegression, images_shape) -> np.ndarray:
    mask = np.array(lr.coef_).reshape(images_shape)
    return mask


def apply_lr(images: list[np.ndarray], labels: list):

    # aplanar imagenes para aplicar regresion logistica
    flatten_images = [ img.flatten() for img in images ]

    lr = LogisticRegression(**lr_kwargs)
    lr.fit(flatten_images, labels, _sample_weight(labels))    

    return lr


if __name__ == "__main__": pass