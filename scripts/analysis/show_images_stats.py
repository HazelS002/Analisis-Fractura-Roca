from visualize.images import show_images, animate_average, animate_images
from visualize.graphs import plot_hists
from analysis.stats import image_mean, image_median, image_std,\
    image_percentile
from data_process.utils import read_images, select_sample
from data_process.clean_images import clean

from ..config import PROCESSED_IMAGES_DIR as images_dir


def main():
    images, names = read_images(images_dir + "aligned-images/")
    q = .95
    # a, b = 120, 150        # Poisson(136)        # esta parece del patron
    a, b = 200, 240          # Poisson(220)    # esta parece del ruiso

    # Analisis de frecuencias de pixeles
    sample = select_sample(images, names, sample_size=9)
    plot_hists(*sample, a, b)    # parecen dos distribuciones


    images = clean(images, copy=False)    # limapiar imagenes


    # # Animaciones
    # animate_average(images, 20)    # Mostrar animacion de promediado
    # animate_images(images, 20)     # Animación de barrido de imagenes


    # Estadisticas pixel a pixel
    stats = [
        (image_mean(images),          "Average Images"),
        (image_median(images),        "Median Images"),
        (image_std(images),           "Std Images"),
        (image_percentile(images, q), f"Percentile {q} Image"),
    ]

    show_images(*zip(*stats), suptitle="Images Stats")    # mostrar

    return



if __name__ == "__main__":
    main()