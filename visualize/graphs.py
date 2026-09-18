import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, binom


from analysis.config import RECONSTRUCTION_COMPONENTS
from analysis.pca import reconstruct
from analysis.stats import estimate_params

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


def plot_hists(images, names, a=0, b=255):
    n_images = len(images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False,
                             sharex=True, sharey=True)

    # Estimar parámetros de todas las imágenes (mismo orden que images)
    params = estimate_params(images, a=a, b=b)

    ks = np.arange(a, b + 1)   # valores de píxel a graficar

    for i, (img, name) in enumerate(zip(images, names)):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]

        # Conteos y total de píxeles en el rango
        counts = np.bincount(img.ravel(), minlength=256)[a:b + 1]
        N = counts.sum()

        # Histograma
        ax.bar(ks, counts, width=1, alpha=0.5, label="Histogram")

        # Parámetros estimados de esta imagen
        lamb, n, p, _ = params[i]

        # PMF Poisson escalada a frecuencias
        ax.plot(ks, N * poisson.pmf(ks, lamb), color="C1", lw=1.5,
                label=rf"Poisson({lamb:.2f})")

        # PMF Binomial escalada a frecuencias
        # Recordar: se ajustó con éxitos = x - a, n = b - a
        exitos = ks - a
        valid = (exitos >= 0) & (exitos <= n)
        ax.plot(ks[valid], N * binom.pmf(exitos[valid], n, p),
                color="C2", lw=1.5, label=f"Binomial({n}, {p:.2f})")

        ax.set_title(name)
        ax.legend()

    # Apagar ejes vacíos
    for j in range(i + 1, n_cols * n_rows):
        r, c = j // n_cols, j % n_cols
        axes[r, c].axis("off")

    plt.show()
    return fig, axes, params


if __name__ == "__main__": pass