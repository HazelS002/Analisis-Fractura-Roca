resize = (1056, 1352)

mf_kwargs = {    # mpdel fit kwargrs
    "epochs": 30,               # Número de ciclos de entrenamiento
    "batch_size": 2,            # Procesar 4 imágenes a la vez
    "validation_split": 0.2,    # Reservar el 20% para validación
    "verbose": 1
}


c_kwargs = {    # compiler kwargs
    "optimizer": "adam",
    "loss": "mse"
}