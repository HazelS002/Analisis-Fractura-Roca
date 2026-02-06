import numpy as np
from src import SAMPLE_DATA_DIR, TRANSFORMATION_MARGIN
from src.load_images import read_images
from src.transform import clean_images
from src.utils import get_stage_images
from sklearn.linear_model import LogisticRegression


def create_random_images(size: tuple[int, int], n_images: int, noise_level:\
                         float = 0.1) -> list[np.ndarray]:
        
    return [ (~(np.random.rand(*size) < noise_level)).astype(np.uint8) * 255\
              for _ in range(n_images) ]



def create_data(shuffle: bool = True) -> tuple[list[np.ndarray], list[int]]:
    
    # leer y limpiar imagenes
    images, _ = read_images(SAMPLE_DATA_DIR + "aligned-images/")
    images = get_stage_images(clean_images(images), "cleaned")
    
    # crear imagenes aleatorias
    fake_images = create_random_images(TRANSFORMATION_MARGIN, 12, 0.2)

    all_images = images + fake_images
    labels = [1]*len(images) + [0]*len(fake_images)

    if shuffle:
        perm = np.random.permutation(len(all_images))
        all_images = [all_images[i] for i in perm]
        labels = [labels[i] for i in perm]

    return all_images, labels

def apply_lr(Images, labels, **lr_kw):
    lr = LogisticRegression(**lr_kw)
    lr.fit([img.flatten() for img in Images], labels)
    return lr

if __name__ == "__main__":    
    from src.visualitation import show_images

    images, labels = create_data(shuffle=True)
    # show_images(images, labels, suptitle="Generated Dataset")

    lr_model = apply_lr(images, labels, max_iter=1000)
    show_images([np.array(lr_model.coef_).reshape(TRANSFORMATION_MARGIN)], ["LR Coefficients"])
    