from data_process.utils import read_images, select_sample
from visualize.images import show_images
from analysis.clustering import fit_km, cluster, keep_cluster
import joblib


from ..config import PROCESSED_IMAGES_DIR as images_dir
from ..config import MODELS_DIR as models_dir


def main():
    a, b = 100, 240    # rango de valores para entrenar km
    images, names = read_images(images_dir + "aligned-images/")

    km = fit_km(images, a=a, b=b)    # ajustar modelo
    joblib.dump(km, models_dir + "km_pixel_clustering.joblib")  # guardar
    # km = joblib.load(models_dir + "km_pixel_clustering.joblib") # leer modelo

    print("Centroids:\t", *km.cluster_centers_, sep=", ")

    test_images, ti_names = select_sample(images, names, 9)
    quantized, label_images = cluster(test_images, km)
    show_images(quantized, ti_names,
                suptitle=f"Quantized Images by KM-centroids in [{a},{b}].")


    kept_cluster = keep_cluster(test_images, label_images, targets=0, a=a, b=b)
    show_images(kept_cluster, ti_names, suptitle="Original values of Cluster 0")

    return

if __name__ == "__main__":
    main()