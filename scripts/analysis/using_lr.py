from visualize.graphs import plot_hists
from data_process.utils import read_images, read_sample
from ..config import PROCESSED_IMAGES_DIR as images_dir


def main():
    # images, names = read_images(images_dir + "aligned-images/")
    images, names = read_sample(images_dir + "aligned-images/", 9)

    a, b = 120, 150
    # a, b = 200, 240
    plot_hists(images, names, a, b)

    return


if __name__ == "__main__":
    main()