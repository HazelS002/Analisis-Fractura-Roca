from matplotlib import pyplot as plt

# configuracion de matplotlib
plt.rcParams['figure.constrained_layout.use'] = True

# paths
DATA_DIR = "./data/"
SAMPLE_DATA_DIR = DATA_DIR + "sample-images/"
RAW_DATA_DIR = DATA_DIR + "raw/"
PROCESSED_DATA_DIR = DATA_DIR + "processed/"

MODELS_DIR = "./models/"

# tamaño de las imagenes
IMAGE_SIZE = (358, 232)  # (width, height)

# tamaño para imagenes rotadas y traslatadas
TRANSFORMATION_MARGIN = (400, 400)  # (width, height)