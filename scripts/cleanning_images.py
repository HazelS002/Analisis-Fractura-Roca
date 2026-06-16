from data_process.clean_images import clean
from data_process.utils import read_images, save_images
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR

def main(repeat_filters):
    images, names = read_images(PROCESSED_IMAGES_DIR + "aligned-images/")
    results = clean(images, repeat_filters)
    save_images(results, names, PROCESSED_IMAGES_DIR + "binary-images/")
    return


if __name__ == "__main__":
    main(8)