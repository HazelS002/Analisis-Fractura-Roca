from dae.autoencoder import set_dae
from dae.utils.helpers import save_model
from .config import MODELS_DIR

def main():
    model = set_dae()
    save_model(model, MODELS_DIR + "dae.keras")
    
    return


if __name__ == "__main__":
    main()