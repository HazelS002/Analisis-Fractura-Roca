from sklearn.cluster import KMeans
import numpy as np

from .config import km_kwargs

def _get_data(images):
    X = np.concatenate([ img.copy().ravel().astype(float) for img in images ])\
        .reshape(-1, 1)
    return X



def fit_km(images):
    km = KMeans(**km_kwargs)
    X = _get_data(images)
    km.fit(X)

    return km


def cluster(images, km):
    n_images = len(images)
    shape = images[0].shape

    X = _get_data(images)
    labels = km.predict(X)
    centers = km.cluster_centers_.ravel()

    quantized = centers[labels]                     # (n_samples,)
    return quantized.reshape(n_images, *shape).astype(np.uint8), labels.reshape(n_images, *shape).astype(np.uint8)


def keep_cluster(images, label_images, target, background=255):
    """
    Mantiene los píxeles originales de las imágenes donde la etiqueta
    es `target`, y pone el resto a `background`.

    Recibe:
        images       : list[np.ndarray] o np.ndarray (n_images, H, W).
                       Imágenes originales.
        label_images : np.ndarray (n_images, H, W) con índices de cluster.
        target       : int. Índice del cluster a conservar.
        background   : int. Valor para el resto (255 = blanco).

    Devuelve:
        np.ndarray (n_images, H, W) uint8.
    """
    images = np.asarray(images)
    label_images = np.asarray(label_images)

    out = np.full_like(images, background, dtype=np.uint8)
    mask = label_images == target
    out[mask] = images[mask]
    return out


if __name__ == "__main__":
    pass
