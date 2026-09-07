import cv2
import numpy as np

from .utils.helpers import _apply_rigid_transform
from .config import circle_kwargs, line_kwargs, wa_kwargs


def fit_ellipses(images: list, names: list, copy: bool = True):
    """
    Permite seleccionar 5 puntos en cada imagen para ajustar una elipse.
    Teclas:
      - Clic izquierdo: añade un punto (máximo 5).
      - 'r' / 'R': reinicia la selección de puntos desde cero.
      - ESC: rechaza la imagen actual.
      - ESPACIO: acepta la imagen con la elipse ajustada.
    """
    accepted_images, new_names = [], []
    ellipses = []
    r_ord = ord('r')

    for img, name in zip(images, names):
        display = img.copy()
        points = []
        wn = f"Select Points ({name})"
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 5:
                    points.append([x, y])
                    cv2.circle(display, (x, y), **circle_kwargs)
                    cv2.imshow(wn, display)

        cv2.setMouseCallback(wn, click_event)
        cv2.imshow(wn, display)

        rejected, restart = False, False

        while True:
            while len(points) < 5:    # seleccion de puntos
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC -> rechazar
                    rejected = True
                    break
                elif key == r_ord:
                    points.clear()
                    display = img.copy()
                    cv2.imshow(wn, display)

            if rejected: break  # sale del bucle exterior

            # mostrar elipse
            pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            ellipse = cv2.fitEllipse(pts)
            cv2.ellipse(display, ellipse, **line_kwargs)
            cv2.imshow(wn, display)
            print(f"Center: {ellipse[0]},\tAxis: {ellipse[1]},\tAngle: {ellipse[2]}")


            while True:    # aceptar, rechazar, reiniciar
                key = cv2.waitKey(1) & 0xFF
                restart = True if key == ord('r') else False

                if key == 32:  # ESPACIO -> aceptar
                    accepted_images.append(img)
                    new_names.append(name)
                    ellipses.append(ellipse)
                    print(f"Image: {name} accepted")
                elif key == 27:  # ESC -> rechazar
                    rejected = True
                elif key == r_ord:    # reiniciar
                    points.clear()
                    display = img.copy()
                    cv2.imshow(wn, display)

                if key in set([32, 27, r_ord]): break
            if not restart: break

        cv2.destroyWindow(wn)
        if rejected: print(f"Image: {name} rejected")

    return accepted_images, new_names, ellipses

def align_by_ellipses(images, ellipses):
    aligned_images = []
    w, h = wa_kwargs["dsize"] 
    
    for img, ellipse in zip(images, ellipses):
        (cx, cy), _, ang = ellipse
        aligned_img = _apply_rigid_transform(img, ang-90, w/2-cx, h/2-cy, (cx, cy))
        aligned_images.append(aligned_img)
    
    return aligned_images