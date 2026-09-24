from data_process.utils import get_data
import numpy as np
from sklearn.mixture import GaussianMixture


def fit_gmm(images, n_componentes, a=0, b=255):
    X = get_data(images, a, b)  # Valores de pixeles en el rango [a,b] shape: (n_sample, n_features)

    gmm = GaussianMixture(n_components=n_componentes, covariance_type='full', random_state=42, verbose=True)
    gmm.fit(X)

    return gmm


if __name__ == "__main__":
    pass