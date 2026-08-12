from data_process.ellipses import fit_ellipses, align_by_ellipses
from data_process.utils import read_images, save_images
from .config import PROCESSED_IMAGES_DIR

def main():
    images_dir = PROCESSED_IMAGES_DIR + "png-images/"
    images, names = read_images(images_dir)
    ellipses = fit_ellipses(images, copy=True)
    aligned = align_by_ellipses(images, ellipses)
    save_images(aligned, names, PROCESSED_IMAGES_DIR + "ellipses-aligned/")

if __name__ == "__main__":
    main()