from data_process.utils import restrict
from sklearn.cluster import KMeans
import numpy as np

from .config import km_kwargs

def _get_data(images, a, b):
    X = np.concatenate([ restrict(img.copy(), a=a, b=b) for img in images ])\
        .reshape(-1, 1)
    return X


def fit_km(images, a=0, b=255):
    km = KMeans(**km_kwargs)
    X = _get_data(images, a=a, b=b)
    km.fit(X)

    return km


def cluster(images, km, a=0, b=255, background=255):
    n_images = len(images)
    shape = images[0].shape

    X = _get_data(images, a=0, b=255)    # rango completo para reconstruir
    mask = (X.ravel() >= a) & (X.ravel() <= b)    # máscara de rango

    labels = km.predict(X)
    centers = km.cluster_centers_.ravel()
    quantized = centers[labels]

    quantized[~mask] = background; labels[~mask] = -1 # valores fuera del rango

    quantized_images = quantized.reshape(n_images, *shape).astype(np.int16)
    label_images = labels.reshape(n_images, *shape).astype(np.int16)

    return quantized_images, label_images


def keep_cluster(images, label_images, targets, background=255):
    """
    Conserva solo los píxeles cuyo cluster esté en `targets`. El resto se
    sustituye por `background`.

    Se asume que `label_images` ya marca con -1 los píxeles fuera del rango
    de interés.

    Recibe:
        images       : np.ndarray (n_images, H, W). Imágenes (originales o cuantizadas).
        label_images : np.ndarray (n_images, H, W). Índices de cluster (-1 = fuera).
        targets      : int o iterable de ints. Cluster(es) a conservar.
        background   : int. Valor para los píxeles descartados (255 = blanco).

    Devuelve:
        out : np.ndarray (n_images, H, W) uint8.
    """
    images = np.asarray(images)
    label_images = np.asarray(label_images)
    if np.isscalar(targets): targets = [targets]

    out = np.full_like(images, background, dtype=np.uint8)
    mask = np.isin(label_images, targets)
    out[mask] = images[mask]
    return out

if __name__ == "__main__":
    pass
