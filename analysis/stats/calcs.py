import numpy as np

def image_mean(images:list[np.ndarray]) -> np.ndarray:
    return np.mean(images, axis=0)


def image_median(images:list[np.ndarray]) -> np.ndarray:
    return np.median(images, axis=0)


def image_std(images: list[np.ndarray]) -> np.ndarray:
    return np.std(images, axis=0)


def image_percentile(images: list[np.ndarray], q=0.95) -> np.ndarray:
    return np.percentile(images, q=q, axis=0)


if __name__ == "__main__": pass