import cv2 as cv2
import numpy as np
from src import TRANSFORMATION_MARGIN
from src.load_images import save_images
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import img_as_float, img_as_ubyte

def _del_area(binaria, area_max=60, conectividad=4):
    # Asegurar formato uint8 (0 y 255)
    if binaria.dtype == bool:
        img = (binaria.astype(np.uint8)) * 255
    else:
        img = binaria.copy()
        if img.max() == 1:
            img = img * 255
    
    
    # Encontrar componentes blancas en la invertida (que eran negras en original)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(img), connectivity=conectividad)
    
    # Máscara donde se conservarán las regiones negras que queremos mantener (las grandes)
    mascara_negra_conservadas = np.zeros_like(img)
    
    # Omitir fondo (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_max:
            # Esta región negra original es lo suficientemente grande, la conservamos
            mascara_negra_conservadas[labels == i] = 255
    
    
    resultado = np.full_like(img, 255)  # fondo blanco
    resultado[mascara_negra_conservadas == 255] = 0  # poner negro donde había regiones negras grandes
    
    # Devolver en el mismo tipo de dato de entrada
    return resultado if binaria.dtype != bool else resultado > 0

def clean_images(images: list[np.ndarray]) -> list[dict]:
    results = []
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(24, 24))

    for img in images:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result_dict = {"original": img.copy()}

        # Mediana
        median_img = img
        for _ in range(8):
            median_img = cv2.medianBlur(median_img, 3)
            median_clahe = clahe.apply(median_img)
            _, median_thresh = cv2.threshold(median_clahe, 150, 255, cv2.THRESH_BINARY)
        result_dict["cleaned"] = _del_area(median_thresh, area_max=90, conectividad=4)


        results.append(result_dict)

    return results
