import cv2 as cv2
import numpy as np
from src import TRANSFORMATION_MARGIN

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


def select_rotation_line(image: np.ndarray) -> list[list[int, int], list[int, int]]:
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            
            if len(points) == 2:
                cv2.line(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

            cv2.imshow("image", img)

    img = image #image.copy()
    cv2.namedWindow("image", cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow("image", img)

    cv2.setMouseCallback("image", click_event) ; cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    if len(points) != 2:
        raise ValueError("Se deben seleccionar exactamente dos puntos.")

    return points

def align(images:list[np.ndarray], lines_points:\
          list[list[int, int], list[int, int]]) -> list[np.ndarray]:
    
    """  """
    global_center = np.array(TRANSFORMATION_MARGIN) // 2
    
    aligned_images = []
    for img, points in zip(images, lines_points):
        center, p2 = points
        angle = np.arctan2(p2[1] - center[1], p2[0] - center[0]) * 180 / np.pi

        transformation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformation_matrix[0, 2] += (global_center[0] - center[0])
        transformation_matrix[1, 2] += (global_center[1] - center[1])
        
        aligned_image = cv2.warpAffine(img, transformation_matrix,
                                       TRANSFORMATION_MARGIN)
        aligned_images.append(aligned_image)

    return aligned_images
    

if __name__ == "__main__":
    """ """
    from src import SAMPLE_DATA_DIR, IMAGE_SIZE
    from src.load_images import read_images
    from src.visualitation import show_images

    ############################# LEER IMAGENES ###############################

    # # Descomentar para probar la limpieza de imagenes
    images, names = read_images(SAMPLE_DATA_DIR + "images/", IMAGE_SIZE)
    # results = clean_images(images)    # aplicar limpieza

    ############################# ALINEAR IMAGENES ############################

    lines_points = [ select_rotation_line(img) for img in images ]
    aligned_images = align(images, lines_points)


    ########################### VISUALIZAR RESULTADOS ##########################

    show_images(images, names)
    show_images(aligned_images, names)