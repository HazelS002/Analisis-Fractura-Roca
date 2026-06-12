nmf_kwargs = {    # paramestros de NMF
    # "n_components": 4,  # el numero de componentes se maneja directamente
    "init": "random",
    "random_state": 42,
    "max_iter": 500,
    "solver": "mu",
    "beta_loss": "kullback-leibler",
    "verbose": 0
}
