import os
import numpy as np
import cv2


def get_imagesstage(results:list[dict], stage:str) -> list[np.ndarray]:
    return [result[stage] for result in results]


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