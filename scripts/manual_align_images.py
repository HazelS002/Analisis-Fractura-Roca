from data_process.align_images import manual_aligner
from data_process.utils import read_images

from .config import PROCESSED_IMAGES_DIR


def main():
    data_dir = PROCESSED_IMAGES_DIR + "png-images/"
    images, names = read_images(data_dir)
    manual_aligner(images, names)
    return


if __name__ == "__main__":
    main()