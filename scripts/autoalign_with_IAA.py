from data_process.align_images import iterative_average_alignment
from data_process.utils import read_images, save_images
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR

def main(iters):
    images_dir = PROCESSED_IMAGES_DIR + "aligned-images/"
    images, names = read_images(images_dir)

    aligned_images, final_average = iterative_average_alignment(images, iters)

    show_images(images, names, suptitle="Original Images")
    show_images(aligned_images, names, suptitle="Aligned Images")
    show_images([final_average], [""], suptitle="Final Average Image")
    return

if __name__ == "__main__":
    main(50)