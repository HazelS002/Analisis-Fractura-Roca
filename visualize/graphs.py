import numpy as np
from matplotlib import pyplot as plt

from .images import show_images

def show_nmf_components(H:np.ndarray, W:np.ndarray,
                        image_shape:tuple[int, int]) -> None:
    components, titles = [], []

    for i in range(H.shape[0]): # reconstruir cada componente
        component_img = H[i, :].reshape(image_shape)
        title = f'Component {i+1}, Average Intensity: {W[:, i].mean():.2f}'
        components.append(component_img)
        titles.append(title)

    show_images(components, titles, suptitle="NMF - Components")
    return

def plot_reconstruction_error(errors:list[float], components:list[int]) -> None:
    plt.plot(components, errors, color='red', marker='o')
    plt.xlabel("Number of components") ; plt.ylabel("Reconstruction error")
    plt.show()
    return


# def fft_visualitation(image, magnitude, phase, reconstructed, peaks=None, components=None):

#     if components is not None:
#         show_images(components, [f"Comp {i+1}" for i in range(len(components))])

#     _, axes = plt.subplots(1, 4)
#     cy, cx = TRANSFORMATION_MARGIN[0]//2, TRANSFORMATION_MARGIN[1]//2

#     axes[0].imshow(image, cmap='gray')
#     axes[0].set_title("Original")
#     axes[0].axis('off')

#     axes[1].imshow(magnitude, cmap='gray', origin='lower')
#     axes[1].set_title("Magnitud")

#     if peaks is not None:
#         for u, v in peaks:
#             axes[1].add_patch(
#                 plt.Circle((cx + v, cy + u), 3, color='red', fill=False)
#             )

#     axes[2].imshow(phase, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
#     axes[2].set_title("Fase")

#     axes[3].imshow(reconstructed, cmap='gray')
#     axes[3].set_title("Reconstrucción")
#     axes[3].axis('off')

#     plt.tight_layout()
#     plt.show()