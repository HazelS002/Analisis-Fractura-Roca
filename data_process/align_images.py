import cv2
import numpy as np
from typing import List, Tuple

from .utils.helpers import save_images, _apply_rigid_transform
from .config import angle, wa_kwargs, circle_kwargs, line_kwargs, wp_kwargs,\
    iterative_average_alignment_tol, min_angle_response, min_desp_response


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
                cv2.line(img, tuple(points[0]), tuple(points[1]), **line_kwargs)
            
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
        rotation_angle = angle + np.arctan2(p2[1] - center[1], p2[0]\
                                            - center[0])*180/np.pi

        dx, dy = global_center[0] - center[0], global_center[1] - center[1]
        aligned_image = _apply_rigid_transform(img, rotation_angle, dx, dy,
                                               global_center)
        
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

################################################################################


################################################################################
#          Auto aligner  (iterative_average_alignment with logpolar)           #
################################################################################

def _estimate_logpolar_rigid_transform(img: np.ndarray, ref: np.ndarray)\
    -> Tuple[float, float, float, Tuple[float, float]]:
    """
    Estima la rotación y traslación entre img y ref usando el método
    log-polar + correlación de fase.
    
    Parámetros:
        img, ref: imágenes en escala de grises (2D, uint8 o float32).
    Retorna:
        angle: rotación en grados (entre -180 y 180).
        dx, dy: desplazamiento en píxeles (subpíxel posible).
        responses: 
    """
    h, w = img.shape
    center = (w // 2, h // 2)
    rho_max = int(round(np.hypot(w, h) / 2))
    img_f, ref_f = np.float32(img), np.float32(ref)    # float32 para FFT
    
    # Transformación log-polar para estimar rotacion
    img_polar = cv2.warpPolar(img_f, center=center, maxRadius=rho_max,
                              **wp_kwargs)
    ref_polar = cv2.warpPolar(ref_f, center=center, maxRadius=rho_max,
                              **wp_kwargs)
    
    # Correlación de fase en el dominio log-polar
    (shift_y, _), angle_response = cv2.phaseCorrelate(ref_polar, img_polar)
    
    # Convertir desplazamiento angular a grados
    angle = (shift_y / w) * 360.0
    
    if angle > 180: angle -= 360    # Ajustar al rango -180..180
    elif angle < -180: angle += 360
        
    # Corregir rotación en img (Rotar img en su centro por -angle)
    img_rot = _apply_rigid_transform(img_f, -angle, 0, 0)
    ref_f = _apply_rigid_transform(ref_f, 0, 0, 0)
    
    # Correlación de fase para traslación entre img_rot y ref
    (dx, dy), desp_respose = cv2.phaseCorrelate(ref_f, img_rot)
    
    return angle, dx, dy, (angle_response, desp_respose)


def iterative_average_alignment(images: List[np.ndarray], n_iter: int = 3)\
    -> Tuple[List[np.ndarray], np.ndarray]:
    """ Alinea un conjunto de imágenes mediante registro iterativo usando
    imagen promedio como referencia.
    Warnning; las imagenes deben tener el mismo tamaño que la salida de
    apply_rigid_transform en utils"""
    
    images_float = [np.float32(img) for img in images]    # float para calculos
    reference = np.median(images_float, axis=0), 0, 0, 0  # referencia inicial
    aligned_batch = images_float          # imagenes alineadas de cada iter 

    print("Aligning images...")
    for iter in range(n_iter):
        print(f"\tIteration {iter+1}/{n_iter}")        
        aligned_this_iter = []
        
        # Alinear cada imagen a la referencia actual
        for i, img in enumerate(aligned_batch):
            angle, dx, dy, (angle_response, desp_response)\
                = _estimate_logpolar_rigid_transform(img, reference)

            if angle_response < min_angle_response and\
                desp_response < min_desp_response:
                angle, dx, dy = (angle, 0, 0) if\
                    min_desp_response < min_angle_response else (0, dx, dy)
                print("Max value response selected for transformation")
            else:
                angle = angle if min_angle_response <= angle_response else 0
                dx, dy = (dx, dy) if min_desp_response <= desp_response\
                    else (0, 0)
            
            img_align = _apply_rigid_transform(img, angle, dx, dy)
            aligned_this_iter.append(img_align)
            print(f"\t\tImage {i}: Angle={angle:.6f}, desp=({dx:.4f},{dy:.4f})")
            print(f"\t\t\tAngle reponse:{angle_response:.6f}")
            print(f"\t\t\tDesp reponse: {desp_response:.6f}")
        
        new_reference = np.mean(aligned_this_iter, axis=0)    # nueva referencia
        
        # cambio relativo respecto a la referencia anterior
        diff = np.linalg.norm(new_reference - reference)\
            /np.linalg.norm(reference + 1e-6)
        print(f"\tRelative change of reference: {diff:.6f}")
        
        if diff < iterative_average_alignment_tol:
            print(f"Convergence at {iter+1} iter")
            break

        reference = new_reference
        aligned_batch = aligned_this_iter
    
    # Convertir las imágenes alineadas de vuelta a uint8
    aligned_uint8 = [np.clip(img, 0, 255).astype(np.uint8)\
                     for img in aligned_batch]
    final_average = np.clip(new_reference, 0, 255).astype(np.uint8)

    return aligned_uint8, final_average


if __name__ == "__main__": pass