import numpy as np
from matplotlib import pyplot as plt


def show_images(images: list, names: list[str], show: bool = True,
                suptitle: str = None, cmap="gray", **fig_kw):
    """Muestra una lista de imágenes."""

    # calcular dimension de malla de imagenes
    n_images = len(images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    
    fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,squeeze=False,**fig_kw)
    if suptitle is not None: fig.suptitle(suptitle)
    
    for i, (img, name) in enumerate(zip(images, names)):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        ax.imshow(img, cmap=cmap) ; ax.set_title(name)
        ax.axis("off")
    
    for j in range(i + 1, n_cols * n_rows):    # Apagar ejes vacíos
        r, c = j // n_cols, j % n_cols
        axes[r, c].axis("off")
    
    if show: plt.show()
    return fig, axes


def show_stages(results:list[dict], show:bool=True, cmap="gray"):
    stages = results[0].keys()
    n_images, n_stages = len(results), len(stages)

    fig, axes = plt.subplots(nrows=n_stages, ncols=n_images)
    if n_images == 1: axes = axes[np.newaxis, :]

    for row, stage in enumerate(stages):
        for col, result in enumerate(results):
            ax = axes[row, col]
            img = result[stage]
            ax.imshow(img, cmap=cmap)
            ax.set_title(stage) ; ax.axis("off")

    if show: plt.show()
    return fig, axes
