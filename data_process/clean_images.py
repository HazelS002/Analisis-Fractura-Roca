import cv2 as cv2
import numpy as np

from .config import min_area, cc_kwargs, clahe_kwargs, mb_kwargs, thresh_kwargs


def _remove_small_areas(binary_img, min_area):
    """
    Elimina componentes blancos pequeños de una imagen binaria.
    
    Parámetros:
        binary_img: imagen binaria (dtype=bool o uint8, 0 y 255)
        min_area: área mínima para conservar un componente blanco
        connectivity: 4 u 8
    Retorna:
        imagen binaria del mismo tipo que la entrada
    """
    # Convertir a uint8 (0 y 255) si es necesario
    if binary_img.dtype == bool:
        img = binary_img.astype(np.uint8) * 255
    else:
        img = binary_img.copy()
        if img.max() == 1: img = img * 255
    
    # Etiquetar componentes blancos (sin invertir)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(\
        img, **cc_kwargs)
    
    result = np.zeros_like(img)    # mascara de areas grandes
    for i in range(1, num_labels):
        # si el area es grande se cambia blanco
        if stats[i, cv2.CC_STAT_AREA] >= min_area: result[labels == i] = 255    
    
    # Devolver en el mismo tipo que la entrada
    if binary_img.dtype == bool: return result.astype(bool)
    elif binary_img.max() == 1: return result // 255
    else: return result


def _apply_filters(images: list[np.ndarray]) -> list[np.ndarray]:
    results = []
    clahe = cv2.createCLAHE(**clahe_kwargs)

    for img in images.copy():
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)\
                .astype(np.uint8)

        # Mediana
        median_img = cv2.medianBlur(img, **mb_kwargs)
        median_clahe = clahe.apply(median_img)
        _, median_thresh = cv2.threshold(median_clahe, **thresh_kwargs)
        without_smallareas = _remove_small_areas(median_thresh, min_area)

        results.append(without_smallareas)

    return results


def clean(images: list[np.ndarray], repeat_filters: int = 1)\
    -> list[np.ndarray]:
    results = images
    for _ in range(repeat_filters): results = _apply_filters(results)

    return results


if __name__ == "__main__": pass