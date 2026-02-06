import cv2
import numpy as np
import os

def read_images(dir:str, image_size=None) -> tuple[list[np.ndarray], list[str]]:
    """ Lee todas las imagenes en formato PNG de un directorio, las redimensiona
    al tamaño especificado y las normaliza a valores entre 0 y 255."""
    images, names = [], []

    for file in os.listdir(dir):            # leer cada archivo en la carpeta
        if file.lower().endswith(".png"):   # si es png
            # Leer imagen en blanco y negro
            print(f"Reading image: {file}")
            image = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)

            # redimencionar imagen
            if image_size is not None:
                print(f"\tOriginal shape:\t{image.shape}")
                image = cv2.resize(image, image_size, interpolation\
                                   =cv2.INTER_AREA)
                print(f"\tNew shape:\t{image.shape}\n")
            else:
                print(f"\tShape:\t{image.shape}\n")

            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)\
                .astype(np.uint8) # normalizar imagen
            
            if image is not None:
                images.append(image) ; names.append(file)
            else:
                raise ValueError(f"Error al leer la imagen: {file}")
    
    return images, names

def save_images(images:list[np.ndarray], names:list[str], dir:str) -> None:
    """ Guarda una lista de imagenes en el directorio especificado con los
    nombres dados."""
    for img, name in zip(images,names): cv2.imwrite(os.path.join(dir,name), img)

if __name__ == "__main__":
    """ Prueba de la función de lectura de imagenes. """
    from src.visualitation import show_images
    from src import SAMPLE_DATA_DIR, IMAGE_SIZE

    images, names = read_images(SAMPLE_DATA_DIR + "images/", IMAGE_SIZE)
    show_images(images, names, suptitle="Imagenes originales")
