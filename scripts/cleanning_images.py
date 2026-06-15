from data_process.clean_images import clean
from data_process.utils import read_images
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR

def main(repeat_filters):
    images, titles = read_images(PROCESSED_IMAGES_DIR + "png-images/")

    n_images = 6
    images, titles = images[:n_images], titles[:n_images]

    results = clean(images, repeat_filters)

    show_images(images, titles, suptitle="Clean Images")
    show_images(results, titles, suptitle="Clean Images")

    return


if __name__ == "__main__":
    main(4)