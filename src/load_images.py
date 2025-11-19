import cv2
import numpy as np
import os
from src import IMAGE_SIZE

def read_images(dir:str, image_size=IMAGE_SIZE) -> tuple[list[np.ndarray], list[str]]:
    images, names = [], []

    for file in os.listdir(dir):
        if file.lower().endswith(".png"):
            # Leer imagen en blanco y negro
            image = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)

            # redimencionar imagen
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)\
                .astype(np.uint8) # normalizar imagen
            
            if image is not None:
                images.append(image) ; names.append(file)
            else:
                print(f"Error al leer la imagen {file}") ; return None
    
    return images, names

if __name__ == "__main__":
    from src.visualitation import show_images
    from src import SAMPLE_DATA_DIR

    images, names = read_images(SAMPLE_DATA_DIR + "images/")
    show_images(images, names)
