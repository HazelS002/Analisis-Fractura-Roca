from matplotlib import pyplot as plt
import numpy as np

# layout de las imagene



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

