from analysis.stats import *
from visualize.images import show_images, animate_average
from data_process.utils import read_images

from .config import PROCESSED_IMAGES_DIR as images_dir

def main():
    images, _ = read_images(images_dir + "aligned-images/")
    q = .95

    animate_average(images)    # Mostrar animacion de promediado

    stats_images = [    # calcular estadisticas por pixeles
        image_mean(images), image_median(images),
        image_std(images), image_percentile(images, q)
    ]
    
    names = [    # Nombres de estadisticas
        "Average Images", "Median Images",
        "Std Images", f"Percentile {q} Image"
    ]

    show_images(stats_images, names, suptitle="Images Stats")    # mostrar
    return

if __name__ == "__main__":
    main()