from src.utils.load_images import read_images
from src.utils.visualitation import *
from src.denoising import *

from matplotlib import pyplot as plt


if __name__ == "__main__":
    plt.rcParams['figure.constrained_layout.use'] = True

    images, names = read_images("./data/sample-images/")
    show_images(images, names)  # Mostrar imagenes originales

    results = clean_images(images)
    show_stages(results)