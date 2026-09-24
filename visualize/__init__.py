import matplotlib.pyplot as plt
from .config import PLOT_CONFIG as pc

from . import utils, images, graphs

# aplicar configuracion
plt.rcParams.update(pc)

__all__ = ["utils", "images", "graphs"]