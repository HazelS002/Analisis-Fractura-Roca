from analysis.stats import *
from visualize.images import show_images
from data_process.utils import read_images

from .config import PROCESSED_IMAGES_DIR

def main():
    images_dir = PROCESSED_IMAGES_DIR + "aligned-images/"
    images, _ = read_images(images_dir)

    show_images([image_mean(images)], [""], suptitle="Average Image")
    show_images([image_median(images)], [""], suptitle="Median Image")
    show_images([image_std(images)], [""], suptitle="Std Image")
    show_images([image_percentile(images)], [""], suptitle\
                =f"Percentile {q} Image")
    
    return

if __name__ == "__main__":
    main()