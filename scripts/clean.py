from data_process.clean_images import clean
from data_process.utils import read_sample
from visualize.images import show_images

from  .config import PROCESSED_IMAGES_DIR as images_dir

def main():
    images, names = read_sample(images_dir + "aligned-images/", sample_size=10)
    results = clean(images, copy=False)

    show_images(results, names)
    return


if __name__ == "__main__":
    main()