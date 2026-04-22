
# paths
DATA_DIR = "./data/"
RAW_DATA_DIR = DATA_DIR + "raw/"
PROCESSED_IMAGES_DIR = DATA_DIR + "processed/"

MODELS_DIR = "./models/"

# tamaño de las imagenes
IMAGE_SIZE = (358, 232)  # (width, height)

# tamaño para imagenes rotadas y traslatadas
TRANSFORMATION_MARGIN = (400, 400)  # (width, height)


NMF_PARAMS = {
    # "n_components": 4, # El numero de componentes se agrega directamente
    "init": "random",
    "random_state": 42,
    "max_iter": 500,
    "solver": "mu",
    "beta_loss": "kullback-leibler",
    "verbose": 0
}
