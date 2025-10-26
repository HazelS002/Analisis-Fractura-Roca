from src.utils.load_images import read_images
from src.utils.visualitation import *
from src.denoising import clean_calcos

from matplotlib import pyplot as plt

if __name__ == "__main__":
    images, names = read_images("./data/sample-images/")
    show_images(images, names)  # Mostrar imagenes originales

    results = clean_calcos(images)
    show_stages(results)


    # results = procesar_calcos(images)

    # mostrar resultados

    # show_stages(results)

