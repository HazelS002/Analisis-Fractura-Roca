import cv2
import numpy as np

from data_process.utils.helpers import _apply_rigid_transform

def fit_ellipses(images: list, copy: bool = True):
    ellipses = []
    
    for idx, img in enumerate(images):
        points = []      # actualizar puntos        
        wn = f"Select Points (img {idx+1})"    # nombre de la ventana
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        display = img if not copy else img.copy()
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 5:
                    points.append([x, y])
                    cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow(wn, display)
        
        cv2.setMouseCallback(wn, click_event)
        cv2.imshow(wn, display)
        
        # Esperar a que se acumulen 5 puntos
        while len(points) < 5:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para abortar
                cv2.destroyWindow(wn)
                ellipses.append(None)
                break
        else:
            # Convertir puntos al formato que espera cv2.fitEllipse: (N,1,2)
            pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            ellipse = cv2.fitEllipse(pts)
            
            # Mostrar la elipse resultante brevemente
            cv2.ellipse(display, ellipse, (0, 0, 255), 5)
            cv2.imshow(wn, display)
            cv2.waitKey(0)  # Pulsa cualquier tecla para cerrar
            cv2.destroyWindow(wn)
            ellipses.append(ellipse)

    return ellipses

def align_by_ellipses(images, ellipses):
    """
    Alinea cada imagen basándose en su elipse correspondiente.
    
    Para cada imagen (no se modifica la original) se aplica una transformación
    rígida (rotación + traslación) tal que:
      - El eje mayor de la elipse queda horizontal (ángulo 0°).
      - El centro de la elipse se sitúa en el centro de la imagen resultante.
    
    Args:
        images:   Lista de imágenes (rutas o arrays numpy).
        ellipses: Lista de elipses en el mismo orden, cada una del formato
                  ((cx, cy), (ancho, alto), angulo) devuelto por cv2.fitEllipse.
                  Puede contener None si la selección se canceló.
    
    Returns:
        Lista de imágenes alineadas (arrays numpy).
    """
    aligned_images = []
    
    for img, ellipse in zip(images, ellipses):
        img = img.copy()

        (cx, cy), (_, _), angle = ellipse # elipse devuelta por cv2.fitEllipse
        h, w = img.shape[:2]

        # Centro destino (mitad de la imagen)
        target_cx, target_cy = w/2.0, h/2.0
        aligned = _apply_rigid_transform(img, angle=-angle,
                                         dx=target_cx-cx, dy=target_cy-cy,
                                         center=(int(cx), int(cy)))
        aligned_images.append(aligned)

    return aligned_images