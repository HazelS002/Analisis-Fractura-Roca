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


def cluster(images, km):
    n_images = len(images)
    shape = images[0].shape

    X = _get_data(images, a=0, b=255)    # no restringimos para reconstruir
    labels = km.predict(X)
    centers = km.cluster_centers_.ravel()

    quantized = centers[labels]                     # (n_samples,)
    quantized_images = quantized.reshape(n_images, *shape).astype(np.uint8)
    label_images = labels.reshape(n_images, *shape).astype(np.uint8)

    return quantized_images, label_images  # (, imagen de etiquetas)


def keep_cluster(images, label_images, targets, a=0, b=255, background=255):
    """
    Conserva solo los píxeles que pertenecen al cluster `target` y cuyo
    valor original está dentro de [a, b]. El resto se sustituye por
    `background`.

    Recibe:
        images          : np.ndarray (n_images, H, W) uint8. Imágenes (originales o cuantizadas).
        label_images    : np.ndarray (n_images, H, W) uint8. Índices de cluster.
        targets         : int o iterable de ints. Cluster(es) a conservar.
        a, b            : int. Rango de valores válidos (inclusive).
        background      : int. Valor para los píxeles descartados (255 = blanco).
        original_values : bool.
                          - True: en los píxeles conservados se usa el valor
                            original de la imagen.
                          - False: se usa el valor del centroide del cluster
                            (requiere pasar ya las imágenes cuantizadas).

    Devuelve:
        out : np.ndarray (n_images, H, W) uint8.
    """
    images = np.asarray(images)
    label_images = np.asarray(label_images)
    if np.isscalar(targets): targets = [targets]

    out = np.full_like(images, background, dtype=np.uint8)
    mask = np.isin(label_images, targets) & ((images >= a) & (images <= b))
    out[mask] = images[mask]

    return out


if __name__ == "__main__":
    pass
