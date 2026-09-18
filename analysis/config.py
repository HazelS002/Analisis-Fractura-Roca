import numpy as np




## PCA config

kw_pca = {
    "n_components": 10
}

RECONSTRUCTION_COMPONENTS = 3



## logistic regression

lr_kwargs = {         # paramestros ajustables de regresion logistica
    "max_iter": 100,
    "random_state": 42,
    "verbose": True
}

rimages_weight = None    # peso de imagenes reales
fimages_weight = None    # peso de imagenes falsas



# KM params

init_centroids = np.array([[68.0], [136.0], [220.0], [241.0]])

km_kwargs = {
    "n_clusters": 4,
    "init": init_centroids, # MatrixLike | ((...) -> Any) | ['k-means++', 'random']
    "n_init": 1,
    "max_iter": 30,
    "tol": 0.0001,
    "verbose": 1,
    "random_state": 42,
    "copy_x": False,       # en nuestro preprocesamiento ya hacemos copy
    "algorithm": "lloyd"    # ['lloyd', 'elkan']
}