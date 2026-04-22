import numpy as np
from skimage.feature import canny
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_ellipse
from skimage.util import img_as_ubyte


def apply_hough(images):
    results = []

    for image in images:
        image = np.asarray(image)

        # Bordes con Canny
        edges = canny(image,sigma=3)

        # Hough de elipses
        # result = hough_ellipse(
        #     edges,
        #     accuracy=20,
        #     threshold=250,
        #     min_size=20,
        #     max_size=100
        # )

        # Imagen para visualizar las elipses detectadas
        # base_img = img_as_ubyte(image)
        # hough_img = np.dstack([base_img, base_img, base_img])

        # if len(result) > 0:
        #     # Ordenar por acumulador
        #     result.sort(order='accumulator')

        #     # Dibujar todas las elipses detectadas, de mejor a peor
        #     for best in result[::-1]:
        #         yc = best["yc"]
        #         xc = best["xc"]
        #         a = best["a"]
        #         b = best["b"]
        #         orientation = best["orientation"]

        #         rr, cc = ellipse_perimeter(
        #             int(round(yc)),
        #             int(round(xc)),
        #             int(round(a)),
        #             int(round(b)),
        #             orientation=orientation,
        #             shape=image.shape
        #         )
        #         hough_img[rr, cc] = (255, 0, 0)  # rojo

        results.append({
            "original": image,
            "canny": edges
            # "hough": hough_img
        })

    return results




if __name__ == "__main__":
    from src.load_images import read_images
    from src.visualitation import show_images, show_stages
    from src import PROCESSED_IMAGES_DIR, IMAGE_SIZE

    images, names = read_images(PROCESSED_IMAGES_DIR + "png-images/", IMAGE_SIZE)
    show_stages(apply_hough(images))

