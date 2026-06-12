import cv2
import numpy as np

from .utils.helpers import save_images
from .config import angle, wa_kwargs, circle_kwargs


################################################################################
#                               manual aligner                                 #
################################################################################

def _select_rotation_line(image: np.ndarray, draw_line:bool=True)\
    -> list[list[int, int], list[int, int]]:
    """ Interfaz grafica para seleccionar linea de rotacion de una imagen """

    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            
            if len(points) == 2:    # cuando se han seleccionado dos puntos
                cv2.line(img, tuple(points[0]), tuple(points[1]),
                         (0, 255, 0), 2)
            
            cv2.circle(img, (x, y), **circle_kwargs)    # marcar punto
            cv2.imshow("image", img)    # actualizar imagen

    img = image.copy() if not draw_line else image
    cv2.namedWindow("image", cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow("image", img)

    cv2.setMouseCallback("image", click_event) ; cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2: raise ValueError("Select two points exactly")

    return points

def _align(images:list[np.ndarray], lines_points:\
           list[list[int, int], list[int, int]]) -> list[np.ndarray]:

    global_center = wa_kwargs["dsize"][0]//2, wa_kwargs["dsize"][1]//2
    
    aligned_images = []
    for img, points in zip(images, lines_points):
        center, p2 = points
        rotation_angle = angle\
            + np.arctan2(p2[1] - center[1], p2[0] - center[0])*180/np.pi

        transformation_matrix = cv2.getRotationMatrix2D(center,
                                                        rotation_angle, 1)
        transformation_matrix[0, 2] += (global_center[0] - center[0])
        transformation_matrix[1, 2] += (global_center[1] - center[1])
        
        aligned_image = cv2.warpAffine(img, transformation_matrix, **wa_kwargs)
        aligned_images.append(aligned_image)

    return aligned_images


def manual_aligner(images:list[np.ndarray], names:list[str] = None,\
                   saving_dir: str = None) -> list[np.ndarray]:
    """  """

    lines_points = [ _select_rotation_line(img, draw_line=(saving_dir is None))\
                    for img in images ]
    aligned_images = _align(images, lines_points)

    # guardar imagenes
    if saving_dir is not None: save_images(aligned_images, names, saving_dir)
    return aligned_images


if __name__ == "__main__": pass