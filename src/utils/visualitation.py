from matplotlib import pyplot as plt
import numpy as np


def show_images(images:list, titles:list[str], show:bool=True):
    """  """
    fig, axes = plt.subplots(ncols=3, nrows=4)
    
    for i, img in enumerate(images):
        ax = axes.flat[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(titles[i]) ; ax.axis("off")

    if show: plt.show()
    return fig, axes

def show_stages(results:list[dict], show:bool=True):
    """  """
    stages = results[0].keys()
    n_imgs, n_cols = len(results), len(stages)

    fig, axes = plt.subplots(n_imgs, n_cols)
    if n_imgs == 1: axes = axes[np.newaxis, :]

    for i, res in enumerate(results):
        for j, stage in enumerate(stages):
            ax = axes[i, j]
            img = res[stage]
            ax.imshow(img, cmap="gray")
            ax.set_title(stage) ; ax.axis("off")
            
    if show: plt.show()
    return fig, axes

