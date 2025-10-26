import cv2
import numpy as np

def prepare_image(img):
    """Convierte a escala de grises y normaliza a uint8."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_clahe(img, clip=3.0, tile=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)


def denoise_image(img, h=15):
    """Aplica Non-Local Means para suavizar ruido sin perder bordes."""
    return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=21)


# ---------------------------------------------------------------
# 4. Estimación del fondo (morfología o desenfoque)
# ---------------------------------------------------------------
def estimate_background(img, kernel_rel=0.05):
    """Usa apertura morfológica con kernel grande para estimar fondo."""
    h, w = img.shape
    k = max(3, int(min(h, w) * kernel_rel))
    if k % 2 == 0: k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return background

def subtract_background(img, background):
    """Resta el fondo y normaliza para resaltar trazos."""
    sub = cv2.subtract(img, background)
    sub_norm = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    return sub_norm


def adaptive_binarization(img, blocksize=35, C=10, inverse=True):
    """Aplica umbral adaptativo para separar trazos del fondo."""
    block = blocksize if blocksize % 2 == 1 else blocksize + 1
    thresh_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    bin_img = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresh_type, block, C)
    return bin_img

def morphological_cleaning(img, open_size=3, close_size=3):
    """Elimina ruido pequeño y une trazos interrumpidos."""
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    return cleaned

def remove_small_components(img, min_area_ratio=0.0005):
    """Elimina manchas o ruido que ocupa áreas muy pequeñas."""
    h, w = img.shape
    area_thresh = int(h * w * min_area_ratio)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    filtered = np.zeros_like(img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
            filtered[labels == i] = 255
    return filtered

def clean_calco(img, params=None):
    """Ejecuta todas las etapas sobre una sola imagen."""
    if params is None:
        params = {}

    img = prepare_image(img)
    clahe = apply_clahe(img, **params.get("clahe", {}))
    denoised = denoise_image(clahe, **params.get("denoise", {}))
    # bg = estimate_background(denoised, **params.get("background", {}))
    # sub = subtract_background(denoised, bg)
    # sub_eq = apply_clahe(sub)  # realce post-sustracción
    # thresh = adaptive_binarization(sub_eq, **params.get("binarization", {}))
    # morph = morphological_cleaning(thresh, **params.get("morph", {}))
    # final = remove_small_components(morph, **params.get("filter", {}))

    return {
        "original": img,
        "clahe": clahe,
        "denoised": denoised #,
        #"background": bg,
        # "sub": sub_eq,
        # "thresh": thresh,
        # "morph": morph,
        # "final": final
    }

def clean_calcos(images):
    return [ clean_calco(img) for img in images]
