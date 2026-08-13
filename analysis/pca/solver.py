from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from data_process.utils.helpers import flatten_images, reshape_images
from .config import kw_pca


def solve(images):
    X = flatten_images(images)

    scaler = StandardScaler()
    pca = PCA(**kw_pca)
    X_pca = pca.fit_transform(scaler.fit_transform(X))

    return scaler, pca, X_pca


def reconstruct(X_pca, pca, scaler, shape, n_components=None):
    if n_components is not None:    # llenar de 0's las demas componentes
        X_pca_limited = np.zeros_like(X_pca)
        X_pca_limited[:, :n_components] = X_pca[:, :n_components]
    else:
        X_pca_limited = X_pca

    inverse = scaler.inverse_transform(pca.inverse_transform(X_pca_limited))
    images = reshape_images(inverse, shape)
    
    return images

if __name__ == "__main__": pass