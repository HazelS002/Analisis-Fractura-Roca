from analysis.logistic_regression import apply_lr, create_data, get_mask
from data_process.utils import read_images
from data_process.config import wa_kwargs
from visualize.images import show_images

from ..config import PROCESSED_IMAGES_DIR as images_dir

def main():
    lam = 220.0                    # parametro de distribución
    images_proportion = 1.0        # clases equilibradas
    shape = wa_kwargs["dsize"][1], wa_kwargs["dsize"][0]

    # preparar datos para ajustar modelo
    images, _ = read_images(images_dir + "aligned-images/")
    train_images, labels = create_data(images, lam, shape, images_proportion)
    show_images(train_images, labels.astype(str), suptitle="Generated DataSet")

    # aplicar el modelo
    lr = apply_lr(train_images, labels)
    pixel_weights = get_mask(lr, images_shape=shape)
    show_images([pixel_weights], ["Pixel Weights"])

    return

if __name__ == "__main__":
    main()