from matplotlib import pyplot as plt
import numpy as np

from ..config import SUBPLOT_DEFAULTS as sp_def


def _axes_grid(n_items):
    n_cols = int(np.ceil(np.sqrt(n_items)))
    n_rows = int(np.ceil(n_items / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, **sp_def)

    flat = axes.ravel()          # lista plana en orden C (fila por fila)
    real = flat[:n_items]        # solo los que se usarán
    empty = flat[n_items:]       # los que se apagarán

    for ax in empty: ax.axis("off")

    return fig, real


if __name__ == "__main__":
    pass