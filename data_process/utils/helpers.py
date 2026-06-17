import os
import numpy as np
import cv2
from typing import Tuple

from ..config import wa_kwargs



def get_lastest(images_dir, format="png") -> int:
    """ Recibe directorio con puras imagenes nombradas por numero y
    devuelve el mayor numero. Si esta vacía devuelve 0.
     - Warning: Formato especifico """

    files = os.listdir(images_dir)
    if files:
        num_images = [ int(file.replace(f".{format}", "")) for file in files ]
    else:
        return 0

    return max(num_images)


def save_images(images:list[np.ndarray], names:list[str], dir:str) -> None:
    """ Guarda una lista de imagenes en el directorio especificado con los
    nombres dados."""
    
    for img, name in zip(images, names):
        cv2.imwrite(os.path.join(dir, name), img)

    return


def read_sample(dir, sample_size, image_size=None, random_state=42):
    np.random.seed(random_state)

    lsdir = os.listdir(dir)
    names = np.random.choice(lsdir, size=sample_size, replace=False)
    images = [cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)\
              for file in names]
    
    if image_size is not None:
        images = [ cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)\
                  for image in images]
        
    return images, names



def read_images(dir:str, image_size=None, format="png")\
    -> tuple[list[np.ndarray], list[str]]:
    """ Lee todas las imagenes en formato PNG de un directorio, las
    redimensiona al tamaño especificado y las normaliza a valores
    entre 0 y 255."""

    images, names = [], []

    for file in os.listdir(dir):    # leer cada archivo en la carpeta
        if file.lower().endswith(f".{format}"):    # si es imagen
            # Leer imagen en blanco y negro
            image = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
            print(f"Read image: {file}")

            if image_size is not None:    # redimencionar imagen
                print(f"\tOriginal shape:\t{image.shape}")
                image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
                print(f"\tNew shape:\t{image.shape}\n")
            else:
                print(f"\tShape:\t{image.shape}\n")

            # nomalizar imagen
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)\
                .astype(np.uint8) # normalizar imagen
            
            if image is not None:
                images.append(image) ; names.append(file)
            else:
                raise ValueError(f"Error al leer la imagen: {file}")
            
    return images, names


def _apply_rigid_transform(img: np.ndarray, angle: float, dx: float, dy: float,
                          center: Tuple[int, int] = None) -> np.ndarray:
    """
    Aplica rotación (alrededor del centro) y traslación a una imagen.
    
    Parámetros:
        img: imagen de entrada (uint8 o float32, 2D o 3D).
        angle: ángulo de rotación en grados (sentido antihorario positivo).
        dx, dy: desplazamiento en píxeles (puede ser subpíxel).
        center: centro de rotación (por defecto el centro de la imagen).
    
    Retorna:
        imagen transformada

    Warnning: Primero rota la imagen en el centro dado y despues la traslada
    """
    h, w = img.shape
    center = center if center is not None else (w//2, h//2)
    
    M_trans = cv2.getRotationMatrix2D(center, angle, 1.0)    # rotacion
    M_trans[0, 2] += dx; M_trans[1, 2] += dy                 # traslacion 
    
    # Aplicar transformación afín
    transformed = cv2.warpAffine(img, M_trans, **wa_kwargs)
    return transformed


if __name__ == "__main__": pass