from data_process.align_images import manual_aligner
from data_process.utils import read_images, save_images
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR


def main():
    data_dir = PROCESSED_IMAGES_DIR + "png-images/"
    images, names = read_images(data_dir)
    aligned_images = manual_aligner(images[:5], names[:5], saving_dir=\
                                    PROCESSED_IMAGES_DIR + "aligned-images/")

    show_images(images[:5], names[:5], suptitle="Original Images")
    show_images(aligned_images[:5], names[:5], suptitle="Aligned Images")

    return


if __name__ == "__main__":
    main()