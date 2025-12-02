from matplotlib import pyplot as plt
import numpy as np


def show_images(images:list, names:list[str], show:bool=True):
    """ Muestra una lista de imagenes. """
    n_images = len(images)
    
    n_cols = np.ceil(np.sqrt(n_images)).astype(int)
    n_rowa = np.ceil(n_images / n_cols).astype(int)

    fig, axes = plt.subplots(nrows=n_rowa, ncols=n_cols)

    for i, (img, name) in enumerate(zip(images, names)):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        ax.imshow(img, cmap="gray")
        ax.set_title(name) ; ax.axis("off")

    # apagar ejes vacíos
    for j in range(i+1, n_cols * n_rowa):
        r, c = j // n_cols, j % n_cols
        axes[r, c].axis("off")

    if show: plt.show()
    return fig, axes


def show_stages(results:list[dict], show:bool=True):
    stages = results[0].keys()
    n_images, n_stages = len(results), len(stages)

    fig, axes = plt.subplots(nrows=n_stages, ncols=n_images)
    if n_images == 1: axes = axes[np.newaxis, :]

    for row, stage in enumerate(stages):
        for col, result in enumerate(results):
            ax = axes[row, col]
            img = result[stage]
            ax.imshow(img, cmap="gray")
            ax.set_title(stage) ; ax.axis("off")

    if show: plt.show()
    return fig, axes


if __name__ == "__main__": pass
