from matplotlib import pyplot as plt
import numpy as np
from src import TRANSFORMATION_MARGIN

plt.rcParams["figure.constrained_layout.use"] = True


def show_images(images: list, names: list[str], show: bool = True,
                suptitle: str = None, **fig_kw):
    """Muestra una lista de imágenes."""
    n_images = len(images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    
    fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,squeeze=False,**fig_kw)
    if suptitle is not None: fig.suptitle(suptitle)
    
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

def plot_reconstruction_error(errors:list[float], components:list[int]) -> None:
    plt.plot(components, errors, color='red', marker='o')
    plt.xlabel("Number of components") ; plt.ylabel("Reconstruction error")
    plt.show()
    return


def fft_visualitation(image, magnitude, phase, reconstructed, peaks=None, components=None):

    if components is not None:
        show_images(components, [f"Comp {i+1}" for i in range(len(components))])

    _, axes = plt.subplots(1, 4)
    cy, cx = TRANSFORMATION_MARGIN[0]//2, TRANSFORMATION_MARGIN[1]//2

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(magnitude, cmap='gray', origin='lower')
    axes[1].set_title("Magnitud")

    if peaks is not None:
        for u, v in peaks:
            axes[1].add_patch(
                plt.Circle((cx + v, cy + u), 3, color='red', fill=False)
            )

    axes[2].imshow(phase, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
    axes[2].set_title("Fase")

    axes[3].imshow(reconstructed, cmap='gray')
    axes[3].set_title("Reconstrucción")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__": pass
