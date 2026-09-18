
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