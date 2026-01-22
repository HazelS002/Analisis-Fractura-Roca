from matplotlib import pyplot as plt
import numpy as np


def show_images(images: list, names: list[str], show: bool = True):
    """Muestra una lista de imágenes."""
    n_images = len(images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    
    for i, (img, name) in enumerate(zip(images, names)):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        ax.imshow(img, cmap="gray") ; ax.set_title(name)
        ax.axis("off")
    
    for j in range(i + 1, n_cols * n_rows):    # Apagar ejes vacíos
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

def show_nmf_components(H:np.ndarray, W:np.ndarray,
                        image_shape:tuple[int, int]) -> None:
    components, titles = [], []

    for i in range(H.shape[0]): # reconstruir cada componente
        component_img = H[i, :].reshape(image_shape)
        title = f'Component {i+1}, Average Intensity: {W[:, i].mean():.2f}'
        components.append(component_img)
        titles.append(title)

    show_images(components, titles)
    return


if __name__ == "__main__": pass
