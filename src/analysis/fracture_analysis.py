from src.utils.load_images import read_images
from src.utils.visualitation import *
from src.denoising import *
from src.denoising.Noise2Void import Noise2Void

from matplotlib import pyplot as plt


if __name__ == "__main__":
    plt.rcParams['figure.constrained_layout.use'] = True

    images, names = read_images("./data/sample-images/images/")
    # show_images(images, names)  # Mostrar imagenes originales

    results = clean_images(images)
    # show_stages([results[6]])

    fig, ax = plt.subplots(ncols=2, nrows=1)

    ax[0].imshow(results[6]["original"], cmap="gray")
    ax[1].imshow(results[6]["thresh"], cmap="gray")
    plt.show()

    # 
    # n2v = Noise2Void()
    # n2v.load_trained_model("./models/Noise2Void/Noise2Void.h5")
    # n2v_images = n2v.denoise(np.array([np.expand_dims(img, axis=-1) for img in images]))

    # show_images(n2v_images, names)

