import cv2 as cv2
import numpy as np

from .config import canny_kwargs, blur_kwargs, ksize_medianB


def _check_format(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)\
            .astype(np.uint8)

    return image


def _apply_canny(image, dilate_kernel=None, iterations=None, **canny_kwargs):
    edges = cv2.Canny(image, **canny_kwargs)

    if dilate_kernel is not None:
        iterations = 1 if iterations is None else iterations
        edges = cv2.dilate(edges, dilate_kernel, iterations=iterations)

    return edges


def _blur(image):
    return cv2.GaussianBlur(255-image, **blur_kwargs)


def _apply(image, copy):
    img = image if not copy else image.copy()    # hacer copia si se requiere
    img = _check_format(img)                     # corregir formato imagen

    if ksize_medianB is not None:
        img = cv2.medianBlur(img, ksize_medianB)


    # aqui se elije que aplicar
    # img = _blur(img)
    img = _apply_canny(img, **canny_kwargs)

    return img


def clean(images: list[np.ndarray], copy: bool=False) -> list[np.ndarray]:
    return [ _apply(img, copy=copy) for img in images ]


if __name__ == "__main__": pass