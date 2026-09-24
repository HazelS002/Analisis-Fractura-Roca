import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, binom


from analysis.config import RECONSTRUCTION_COMPONENTS
from analysis.pca import reconstruct
from analysis.stats import estimate_params
from analysis.clustering import cluster

from visualize.images import show_images
from visualize.utils import _axes_grid
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


def simple_hists(images, names, a=0, b=255, density=True, show=True):
    n_images = len(images)
    fig, axes = _axes_grid(n_images)

    bin_edges = np.arange(a - 0.5, b + 1.5)

    for img, name, ax in zip(images, names, axes):
        ax.hist(img.ravel(), bins=bin_edges, alpha=0.5,\
            label="Intensity frequency", density=density)

        ax.set_title(name)
        ax.legend()

    if show: plt.show()

    return fig, axes


def plot_hists(images, names, km, a=0, b=255):
    n_images = len(images)
    fig, axes = _axes_grid(n_images)

    centers = km.cluster_centers_.ravel()
    ks = np.arange(a, b + 1)
    bin_edges = np.arange(a - 0.5, b + 1.5)

    _, labels = cluster(images, km, a=a, b=b)

    for i, (img, name) in enumerate(zip(images, names)):
        flat = np.asarray(img).ravel()

        for k_cluster, lam in enumerate(centers):
            mask = labels[i].ravel() == k_cluster

            if not mask.any(): continue

            axes[i].hist(flat[mask], bins=bin_edges, alpha=0.5,\
                color=f"C{k_cluster}", label=f"Cluster {k_cluster}")
            
            axes[i].plot(ks, mask.sum() * poisson.pmf(ks, lam),\
                color=f"C{k_cluster}", lw=1.5, label=rf"Poisson($\lambda$={lam:.1f})")

        axes[i].set_title(name)
        axes[i].legend(fontsize=6)

    plt.show()
    return fig, axes
        

def show_gmm(gmm_model, images, names, a=0, b=255):

    fig, axes = simple_hists(images, names, a=a, b=b, density=True, show=False)

    x_vals = np.linspace(a, b, 1000).reshape(-1, 1)
    log_prob = gmm_model.score_samples(x_vals)
    pdf = np.exp(log_prob)

    for ax in axes:
        ax.plot(x_vals, pdf, '-k', linewidth=2, label='GMM Total')

        for i in range(gmm_model.n_components):
            mean = gmm_model.means_[i][0]
            cov = gmm_model.covariances_[i][0][0]
            weight = gmm_model.weights_[i]
            
            # PDF de una Gaussiana individual
            component_pdf = weight * (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-0.5 * ((x_vals.ravel() - mean)**2) / cov)
            ax.plot(x_vals, component_pdf, '--', label=f'Componente {i+1} (Media: {mean:.1f})')

        ax.legend(fontsize=4)

    plt.show()
    return


if __name__ == "__main__": pass