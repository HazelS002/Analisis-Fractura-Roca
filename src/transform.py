import cv2 as cv2
import numpy as np

def clean_images(images:list[np.ndarray]) -> list[dict[str, np.ndarray]]:
    results = []
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))

    for img in images:
        nlmeans_img = cv2.fastNlMeansDenoising(img, None, h=15,\
                        templateWindowSize=11, searchWindowSize=21)
        
        # aplicar contraste
        clahe_img = clahe.apply(nlmeans_img)

        # aplicar humbral
        _, thresh_img = cv2.threshold(clahe_img, 55, 255, cv2.THRESH_BINARY)

        results.append({
            "original": img,
            "cleaned": thresh_img
        })

    return results



# vamos a trabajar de la siguiente manera, seleccionamos los centros manualmente

def select_centers(images: list[np.ndarray]) -> list[list[int, int]]:
    centers = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            centers.append([x, y])
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("image", img)

    for img in images:
        img = img.copy()
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return centers

# seleccionar dos puntos (estos definen un recta), al colocar vertical la
# recta se rota la imagen 

def select_rotation_line(image: np.ndarray) -> list[list[int, int], list[int, int]]:
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            
            if len(points) == 2:
                cv2.line(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
            
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("image", img)

    img = image.copy()
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        raise ValueError("Se deben seleccionar exactamente dos puntos.")

    return points

    

if __name__ == "__main__":
    """ Prueba de la función de limpieza de imagenes y
    visualización de etapas. """
    from src import SAMPLE_DATA_DIR, IMAGE_SIZE
    from src.load_images import read_images
    from src.visualitation import show_images

    ############################# LEER IMAGENES ###############################

    # # Descomentar para probar la limpieza de imagenes
    images, names = read_images(SAMPLE_DATA_DIR + "images/", IMAGE_SIZE)
    # results = clean_images(images)    # aplicar limpieza

    ########################### VISUALIZAR RESULTADOS ##########################

    centers = select_centers(images)
    # lines = [ select_rotation_line(img) for img in images]

    

    