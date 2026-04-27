import cv2 as cv2
import numpy as np
from src import TRANSFORMATION_MARGIN
from src.load_images import save_images
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import img_as_float, img_as_ubyte

def del_area(binaria, area_max=60, conectividad=4):
    # Asegurar formato uint8 (0 y 255)
    if binaria.dtype == bool:
        img = (binaria.astype(np.uint8)) * 255
    else:
        img = binaria.copy()
        if img.max() == 1:
            img = img * 255
    
    
    # Encontrar componentes blancas en la invertida (que eran negras en original)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(img), connectivity=conectividad)
    
    # Máscara donde se conservarán las regiones negras que queremos mantener (las grandes)
    mascara_negra_conservadas = np.zeros_like(img)
    
    # Omitir fondo (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_max:
            # Esta región negra original es lo suficientemente grande, la conservamos
            mascara_negra_conservadas[labels == i] = 255
    
    
    resultado = np.full_like(img, 255)  # fondo blanco
    resultado[mascara_negra_conservadas == 255] = 0  # poner negro donde había regiones negras grandes
    
    # Devolver en el mismo tipo de dato de entrada
    return resultado if binaria.dtype != bool else resultado > 0

def clean_images(images: list[np.ndarray]) -> list[dict]:
    results = []
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(24, 24))

    for img in images:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result_dict = {"original": img.copy()}

        # Mediana
        median_img = img
        for _ in range(8):
            median_img = cv2.medianBlur(median_img, 3)
            median_clahe = clahe.apply(median_img)
            _, median_thresh = cv2.threshold(median_clahe, 150, 255, cv2.THRESH_BINARY)
        result_dict["cleaned"] = del_area(median_thresh, area_max=90, conectividad=4)


        results.append(result_dict)

    return results

def select_rotation_line(image: np.ndarray, draw_line:bool=True)\
    -> list[list[int, int], list[int, int]]:
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            
            if len(points) == 2:
                cv2.line(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

            cv2.imshow("image", img)

    img = image.copy() if not draw_line else image
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
                                       TRANSFORMATION_MARGIN,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
        aligned_images.append(aligned_image)

    return aligned_images


def transform(images:list[np.ndarray], names:list[str] = None,
              saving_dir: str = None) -> list[np.ndarray]:
    """  """

    lines_points = [ select_rotation_line(img, draw_line=(saving_dir is None))\
                    for img in images ]
    aligned_images = align(images, lines_points)

    # guardar imagenes
    if saving_dir is not None: save_images(aligned_images, names, saving_dir)
    return aligned_images
    

if __name__ == "__main__":
    """ """
    from src import PROCESSED_IMAGES_DIR, IMAGE_SIZE
    from src.load_images import read_images
    from src.visualitation import show_images, show_stages
    from src.utils import get_stage_images

    # for_saving = True

    # leer imagenes
    images, names = read_images(PROCESSED_IMAGES_DIR + "png-images/", IMAGE_SIZE)

    # # alinear imagenes
    # param_names, saving_dir = (names, PROCESSED_IMAGES_DIR + "aligned-images/")\
    #     if for_saving else (None, None)
    # aligned_images = transform(images, names=param_names, saving_dir=saving_dir)
    
    # # mostrar imagenes
    # show_images(aligned_images, names, suptitle="Aligned Images")

    # show_stages(clean_images(images[12:20]))

    show_images(get_stage_images(clean_images(images), "cleaned"), names)
    
