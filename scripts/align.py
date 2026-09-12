from data_process.ellipses import fit_ellipses, align_by_ellipses
from data_process.utils import read_images, save_images

from .config import PROCESSED_IMAGES_DIR
import os

def main():
    images_dir = PROCESSED_IMAGES_DIR + "png-images/"
    images, names = read_images(images_dir)
    images, names, ellipses = fit_ellipses(images, names, copy=True)
    aligned = align_by_ellipses(images, ellipses)

    output_dir = PROCESSED_IMAGES_DIR + "aligned-images/"
    [ os.remove(os.path.join(output_dir, file))\
     for file in os.listdir(output_dir)]
    save_images(aligned, names, output_dir)

if __name__ == "__main__":
    main()