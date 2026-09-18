from analysis.stats import *
from visualize.images import show_images, animate_average, animate_images
from visualize.graphs import plot_hists
from data_process.utils import read_images, select_sample

from data_process.clean_images import clean

from ..config import PROCESSED_IMAGES_DIR as images_dir

def main():
    images, names = read_images(images_dir + "aligned-images/")
    q = .95

    images = clean(images, copy=False)

    sample = select_sample(images, names, sample_size=8)
    plot_hists(*sample)

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