import numpy as np
from data_process.utils import restrict


def image_mean(images:list[np.ndarray]) -> np.ndarray:
    return np.mean(images, axis=0)


def image_median(images:list[np.ndarray]) -> np.ndarray:
    return np.median(images, axis=0)


def image_std(images: list[np.ndarray]) -> np.ndarray:
    return np.std(images, axis=0)


def image_percentile(images: list[np.ndarray], q=0.95) -> np.ndarray:
    return np.percentile(images, q=q, axis=0)


def _poisson_params(x):
    lambda_hat = np.mean(x)

    return lambda_hat


def _binomial_params(x, a, b):
    n = b - a
    p_hat = np.mean(x-a) / n

    return n, p_hat

def estimate_params(images, a=0, b=255):
    """
    Estima lambda (Poisson) y (n, p) (Binomial) usando solo píxeles en [a, b].

    Recibe:
        images : list[np.ndarray]
        a, b   : int, límites del rango

    Devuelve:
        params : np.ndarray de shape (len(images), 4)
                 columnas: lambda, n, p, mediana
    """

    params = []

    for img in images:
        x = restrict(img, a, b)
        lamb_hat = _poisson_params(x)
        n, p = _binomial_params(x, a, b)
        median = np.median(x)
        print(f"Poisson:  lamb - {lamb_hat:.5f}; Binomial: n - {n},", end=" ")
        print(f"p - {p:.5f}; Median - {median}; Asymmetry - {lamb_hat-median:.5f}")

        params.append((lamb_hat, n, p, median))
    
    return np.array(params)

if __name__ == "__main__": pass