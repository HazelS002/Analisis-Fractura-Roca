from data_process.ellipses import fit_ellipses, align_by_ellipses
from data_process.utils import read_sample
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR

def main():
    images_dir = PROCESSED_IMAGES_DIR + "png-images/"
    images, names = read_sample(images_dir, 6)

    ellipses = fit_ellipses(images, copy=False)
    aligned = align_by_ellipses(images, ellipses)

    show_images(aligned, names)
                            
if __name__ == "__main__":
    main()