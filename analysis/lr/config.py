lr_kwargs = {         # paramestros ajustables de regresion logistica
    "max_iter": 100,
    "random_state": 42,
    "verbose": True
}

noise_level = 0.05             # nivel de ruido en imagesnes falsas   
fakeimages_proportion = 1.0    # clases equilibradas
sample_weight = None           # para fit en LogisticRegression