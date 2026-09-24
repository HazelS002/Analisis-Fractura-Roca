import os

PLOT_CONFIG = {
    'figure.constrained_layout.use': True,
    'figure.dpi': int(os.getenv('MPL_DPI', 150)),
    # 'figure.figsize': (10, 6),

    # Ejes
    'axes.grid': False,
    'axes.labelsize': 6,
    'axes.titlesize': 8,
    'axes.titleweight': 'bold',

    # Líneas
    'lines.linewidth': 2,

    # Fuentes
    'font.size': 11,
    'font.family': 'sans-serif',

    # Guardado
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',

    
    # legend
    'legend.loc': 'best',              # Posición automática ('best', 'upper right', 'lower left', etc.)
    'legend.fontsize': 4,              # Tamaño de la fuente de la leyenda
    'legend.title_fontsize': 11,       # Tamaño de la fuente del título de la leyenda
    'legend.frameon': False,           # Si dibuja el marco alrededor de la leyenda
    'legend.framealpha': 0.6,          # Transparencia del marco (0 = transparente, 1 = opaco)
    'legend.facecolor': 'white',       # Color de fondo de la leyenda
    'legend.edgecolor': '0.8',         # Color del borde del marco
    'legend.fancybox': True,           # Esquinas redondeadas del marco
    'legend.shadow': True,             # Sombra detrás de la leyenda
    # 'legend.ncol': 1,                  # Número de columnas
    'legend.numpoints': 1,             # Número de puntos en las entradas de tipo línea
    'legend.scatterpoints': 1,         # Número de puntos en las entradas de tipo scatter
    'legend.markerscale': 1.0,         # Escala de los marcadores en la leyenda
    'legend.handlelength': 2.0,        # Longitud de las líneas de la leyenda (en unidades de fuente)
    'legend.handleheight': 0.7,        # Altura de los manejadores de la leyenda
    'legend.handletextpad': 0.8,       # Espacio entre el manejador y el texto
    'legend.borderpad': 0.4,           # Espacio interno entre el borde y el contenido
    'legend.labelspacing': 0.5,        # Espacio vertical entre entradas
    'legend.borderaxespad': 0.5,       # Espacio entre la leyenda y los ejes
    'legend.columnspacing': 2.0,       # Espacio entre columnas (si ncol > 1)
}

SUBPLOT_DEFAULTS = {
    'sharex': True,
    'sharey': True,
    'squeeze': False,
}