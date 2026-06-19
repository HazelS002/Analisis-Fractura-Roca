import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


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


def animate_average(images, delay=100):
    """
    Muestra una animación del promedio acumulado de una lista de imágenes.

    Parámetros:
        images (list of numpy.ndarray): Lista de imágenes (mismo tamaño y tipo).
        delay (int): Tiempo en milisegundos entre cada paso (por defecto 100).
    """
    fig, ax = plt.subplots()
    
    avg = images[0].astype(np.float64)    # primera imagen como promedio inicial
    im = ax.imshow(avg.astype(np.uint8), cmap='gray')
    ax.set_title('Acumulated average (frame 1)')

    def update(frame):
        nonlocal avg

        img = images[frame]    # actualizar promedio
        avg = (avg * frame + img.astype(np.float64)) / (frame + 1)
        
        im.set_data(avg.astype(np.uint8))    # Actualizar imagen
        ax.set_title(f'Acumulated average (frame {frame+1})')
        return im,

    # Detener la animación si se pulsa 'q'
    def on_key(event):
        if event.key == 'q': anim.event_source.stop()
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Crear la animación: frames desde 1 hasta el último índice
    anim = FuncAnimation(
        fig, update,
        frames=range(1, len(images)),
        interval=delay,
        repeat=True,
        blit=False  # False para actualizar el título sin problemas
    )

    plt.show()