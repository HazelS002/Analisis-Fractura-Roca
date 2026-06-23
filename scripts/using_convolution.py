from data_process.utils import read_sample
from data_process.convolution import smoothing
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR as images_dir

def main():
    n_images = 10
    images, names = read_sample(images_dir + "aligned-images/", n_images)
    smoothed = smoothing(images)

    show_images(smoothed, names)
    return

if __name__ == "__main__":
    main()