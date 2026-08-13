import numpy as np
from matplotlib import pyplot as plt

from analysis.pca.config import RECONSTRUCTION_COMPONENTS
from analysis.pca.solver import reconstruct

from visualize.images import show_images
from data_process.utils.helpers import reshape_images

def _plot_variance(pca):
    explained_var = pca.explained_variance_ratio_ # varianza explicada por comp
    n = len(explained_var)                        # numero de componentes
    cumulative_var = np.cumsum(explained_var)     # vars acumuladas
    components = np.arange(1, n + 1)              # componentes

    _, ax = plt.subplots()
    ax.plot(components, explained_var, c="blue", marker="o",
            label="Individual Explained Variance")
    ax.plot(components, cumulative_var, c="red", marker="o",
            label="Acumulated Explained Variance")

    plt.xlabel("Components"); plt.ylabel("Explained Variance"); ax.grid(True)
    plt.legend()
    plt.suptitle("Explained Variance PCA")
    plt.show()

    return

def _plot_components(pca, shape, n_components=None):
    components = pca.components_
    n = len(components) if n_components is None else n_components
    components = [reshape_images(components[c], shape) for c in range(n)]

    show_images(components, [f"Component {c}" for c in np.arange(1, n+1)],
                suptitle="Components of PCA")
    return


def _plot_reconstruction(X_pca, pca, scaler, shape, names):
    reconstructed_images = reconstruct(X_pca, pca, scaler, shape,
                                       n_components=RECONSTRUCTION_COMPONENTS)

    show_images(reconstructed_images, names,
                suptitle="Reconstructed images by PCA"\
                        + f"({RECONSTRUCTION_COMPONENTS} components)")
    return


def plot_pca(X_pca, pca, scaler, shape, names):
    _plot_variance(pca)
    # _plot_components(pca, shape, n_components=RECONSTRUCTION_COMPONENTS)
    _plot_reconstruction(X_pca, pca, scaler, shape, names)

    return

if __name__ == "__main__": pass