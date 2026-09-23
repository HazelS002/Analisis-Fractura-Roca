from data_process.utils import read_sample
from visualize.graphs import simple_hists, plot_hists
from joblib import load

from ..config import PROCESSED_IMAGES_DIR as images_dir
from ..config import MODELS_DIR as models_dir



def main():
    sample_size = 9
    b = 250

    images, names = read_sample(images_dir + "aligned-images/", 9)
    simple_hists(images, names, b=b)


    km = load(models_dir + "km_pixel_clustering.joblib")
    plot_hists(images, names, km, a=0, b=b)

    return


if __name__ == "__main__":
    main()