from analysis.stats import *
from visualize.images import show_images, animate_average, animate_images
from data_process.utils import read_images

from ..config import PROCESSED_IMAGES_DIR as images_dir

def main():
    images, _ = read_images(images_dir + "ellipses-aligned/")
    q = .95

    animate_average(images, 20)    # Mostrar animacion de promediado
    animate_images(images, 20)

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