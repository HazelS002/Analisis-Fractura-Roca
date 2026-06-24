from dae.utils.helpers import load_model
from dae.autoencoder import predict
from data_process.utils.helpers import read_sample
from visualize.images import show_images

from .config import MODELS_DIR
from .config import PROCESSED_IMAGES_DIR as images_dir


def main(sample):

    model = load_model(MODELS_DIR + "dae.keras")
    images, names = read_sample(images_dir + "aligned-images/",
                                sample_size=sample)

    denoissed = predict(model, images)
    show_images(denoissed, names)
    
    return

if __name__ == "__main__":
    main(10)