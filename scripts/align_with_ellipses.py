from data_process.ellipses import fit_ellipses
from data_process.utils import read_sample

from .config import PROCESSED_IMAGES_DIR
def main():
    images_dir = PROCESSED_IMAGES_DIR + "png-images/"
    images, _ = read_sample(images_dir, 6)

    ellipses = fit_ellipses(images)
    print(ellipses)

if __name__ == "__main__":
    main()