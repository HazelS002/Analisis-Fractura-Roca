from data_process.utils import read_images, select_sample
from visualize.images import show_images
from analysis.clustering import fit_km, cluster, keep_cluster
import joblib


from ..config import PROCESSED_IMAGES_DIR as images_dir
from ..config import MODELS_DIR as models_dir



def main():
    images, names = read_images(images_dir + "aligned-images/")

    # km = fit_km(images)                                         # ajustar modelo
    # joblib.dump(km, models_dir + "km_pixel_clustering.joblib")  # guardar
    km = joblib.load(models_dir + "km_pixel_clustering.joblib") # leer modelo


    print(km.cluster_centers_)

    test_images, t_names = select_sample(images, names, 9)
    clustered, label_images = cluster(test_images, km)
    show_images(clustered, t_names)


    kept_cluster = keep_cluster(test_images, label_images, target=1)
    show_images(kept_cluster, t_names)

    return

if __name__ == "__main__":
    main()