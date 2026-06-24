from dae.autoencoder import train_dae
from dae.utils.helpers import load_model
from data_process import read_images


from .config import PROCESSED_IMAGES_DIR as images_dir
from .config import MODELS_DIR

def main():
    images, _ = read_images(images_dir + "aligned-images/")
    model = load_model(MODELS_DIR + "dae.keras")

    train_dae(model, images)
    model.save(MODELS_DIR + "dae.keras")
    return

if __name__ == "__main__":
    main()