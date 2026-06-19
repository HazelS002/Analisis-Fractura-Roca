import numpy as np
from matplotlib import pyplot as plt
import cv2


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


def show_hist(images, names, suptitle):

    fig, axes = plt.subplots(1, len(images), sharey=True)

    for image, name, ax in zip(images, names, axes):
        ax.hist(image.ravel(), bins=256, range=(0, 254), color='black')

    fig.suptitle(suptitle)
    plt.show()

    return fig, axes


def animate_average(images, delay=500):
    """
    Muestra una animación del promedio acumulado de una lista de imágenes.

    Parámetros:
        images (list of numpy.ndarray): Lista de imágenes (mismo tamaño y tipo).
        delay (int): Tiempo en milisegundos entre cada paso (por defecto 500).
    """

    # Inicializar el promedio con la primera imagen
    avg = images[0].astype(np.float64)
    current_frame = avg.astype(np.uint8)

    cv2.namedWindow('Acumulated average', cv2.WINDOW_NORMAL)

    # Mostrar la primera imagen
    cv2.imshow('Acumulated average', current_frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return

    # Recorrer el resto de imágenes
    for i, img in enumerate(images[1:], start=2):
        # Actualizar promedio: avg = (avg * (i-1) + img) / i
        avg = (avg * (i-1) + img.astype(np.float64)) / i
        current_frame = avg.astype(np.uint8)

        cv2.imshow('Promedio acumulado', current_frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
