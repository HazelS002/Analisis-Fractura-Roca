from analysis.pca import solve, reconstruct
from analysis.pca.config import RECONSTRUCTION_COMPONENTS
from data_process import read_images, reshape_images
from visualize.images import show_images
from visualize.graphs import plot_pca

from ..config import PROCESSED_IMAGES_DIR as images_dir

def main():
    images, names = read_images(images_dir + "ellipses-aligned/") # cargar imgs
    scaler, pca, X_pca = solve(images)    # aplicar pca

    plot_pca(X_pca, pca, scaler, images[0].shape, names)

    return

if __name__ == "__main__":
    main()