import numpy as np
from sklearn.linear_model import LogisticRegression
from .config import lr_kwargs, fakeimages_proportion,\
    noise_level, sample_weight



def _create_random_images(shape: tuple[int, int], n_images: int,\
                          noise_level:float) -> list[np.ndarray]:
        
    return [ (~(np.random.rand(*shape) < noise_level))\
            .astype(np.uint8) * 255 for _ in range(n_images) ]


def _create_data(images: list[np.ndarray], noise_level:float,
                 fakeimages_proportion: float = 1.0)\
                    -> tuple[list[np.ndarray], list[int]]:
    
    # crear imagenes aleatorias
    n_fakeimages = int(np.round(fakeimages_proportion*len(images)))
    fake_images = _create_random_images(images[0].shape, n_fakeimages,\
                                        noise_level)

    # concatenar y etiquetar imagenes
    all_images = images + fake_images
    labels = [1]*len(images) + [0]*n_fakeimages

    # permutar imagenes
    perm = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in perm]
    labels = [labels[i] for i in perm]

    return all_images, labels


def apply_lr(images: list[np.ndarray], show_generated: bool=False):
    images_shape = images[0].shape

    print("Creating data...")
    real_and_fake_images, targets =\
        _create_data(images, noise_level, fakeimages_proportion)

    if show_generated:
        from visualize.images import show_images
        show_images(real_and_fake_images, targets, suptitle="Generated DataSet")

    # aplanar imagenes para aplicar regresion logistica
    flatten_images = [ img.flatten() for img in real_and_fake_images ]

    lr = LogisticRegression(**lr_kwargs); print("Applying LR...")
    lr.fit(flatten_images, targets, sample_weight)

    # calcular mascara de pesos
    mask = np.array(lr.coef_).reshape(images_shape)

    return mask, lr


if __name__ == "__main__": pass