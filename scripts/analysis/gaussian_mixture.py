from analysis.gmm import fit_gmm
from data_process.utils import read_images, read_sample
from visualize.graphs import show_gmm
import joblib

from ..config import PROCESSED_IMAGES_DIR as images_dir
from ..config import MODELS_DIR as models_dir

def main():
    n_componentes = 2
    a, b = 0, 250

    # images, names = read_images(images_dir + "aligned-images/")
    # gmm = fit_gmm(images, n_componentes, a=a, b=b)
    # joblib.dump(gmm, models_dir + "gmm.joblib")    # guardar

    images, names = read_sample(images_dir + "aligned-images/", sample_size=9)
    gmm = joblib.load(models_dir + "gmm.joblib")   # leer modelo

    print("Components weights (pixels proportion):", gmm.weights_)
    print("Means:                                 ", gmm.means_.flatten())
    print("Vars:                                  ", gmm.covariances_.flatten())

    show_gmm(gmm, images, names, a=a, b=b)

    return


if __name__ == "__main__":
    main()