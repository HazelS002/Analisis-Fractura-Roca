from analysis.lr import apply_lr
from data_process.utils import read_images
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR


def main():
    images, _ = read_images(PROCESSED_IMAGES_DIR + "binary-images/")
    mask, _ = apply_lr(images)

    show_images([mask], [""], suptitle="Mask of weights pixel's in LR")

    return

if __name__ == "__main__":
    main()