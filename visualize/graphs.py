import numpy as np
from matplotlib import pyplot as plt

from .images import show_images

def plot_reconstruction_error(errors:list[float], components:list[int]) -> None:
    plt.plot(components, errors, color='red', marker='o')
    plt.xlabel("Number of components") ; plt.ylabel("Reconstruction error")
    plt.show()
    return