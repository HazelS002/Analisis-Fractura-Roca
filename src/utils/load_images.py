import cv2
import numpy as np
import os

def read_images(dir):
    images, names = [], []

    for file in os.listdir(dir):
        if file.lower().endswith(".png"):
            # Leer imagen en blanco y negro
            image = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)

            # normalizar imagen
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)\
                .astype(np.uint8)
            
            if image is not None:
                images.append(image) ; names.append(file)
            else:
                print(f"Error al leer la imagen {file}") ; return None
    
    return images, names