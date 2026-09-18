from data_process.utils import read_images, select_sample
from visualize.images import show_images
from analysis.clustering import fit_km, cluster, keep_cluster


from ..config import PROCESSED_IMAGES_DIR as images_dir



def main():
    images, names = read_images(images_dir + "aligned-images/")
    km = fit_km(images)

    print(km.cluster_centers_)

    test_images, t_names = select_sample(images, names, 9)
    clustered, label_images = cluster(test_images, km)
    show_images(clustered, t_names)


    kept_cluster = keep_cluster(test_images, label_images, target=1)
    show_images(kept_cluster, t_names)

    return

if __name__ == "__main__":
    main()