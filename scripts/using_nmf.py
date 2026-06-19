from analysis.nmf import apply_nmf
from data_process.utils import read_images
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR as images_dir


def main(components):
    images, names = read_images(images_dir + "aligned-images/")
    reconstructed_images, _, _ = apply_nmf(images, components)

    show_images(reconstructed_images, names,\
                suptitle=f"Reconstructed images whit NMF, {components} comps")
    
    return


if __name__ == "__main__":
    main(2)