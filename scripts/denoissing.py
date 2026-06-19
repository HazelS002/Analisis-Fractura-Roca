from data_process.clean_images import denoise
from data_process.utils import read_sample
from visualize.images import show_images

from .config import PROCESSED_IMAGES_DIR as images_dir

def main():
    sample_size = 5
    sigma = 1.0

    images, names = read_sample(images_dir, sample_size=sample_size)
    denoised_images = denoise(images, sigma=sigma)

    show_images(denoised_images, names)
    
    return

if __name__ == "__main__":
    main()