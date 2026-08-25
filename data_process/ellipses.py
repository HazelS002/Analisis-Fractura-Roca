import cv2
import numpy as np

from .utils.helpers import _apply_rigid_transform
from .config import circle_kwargs, line_kwargs


def fit_ellipses(images: list, names: list, copy: bool = True):
    accepted_images, new_names = [], []
    ellipses = []
    
    for img, name in zip(images, names):
        points = []
        wn = f"Select Points ({name})"
        
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        
        display = img.copy() if copy else img
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 5:
                    points.append([x, y])
                    cv2.circle(display, (x, y), **circle_kwargs)
                    cv2.imshow(wn, display)
        
        cv2.setMouseCallback(wn, click_event)
        cv2.imshow(wn, display)

        while len(points) < 5:    # esperar 5 puntos
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC: rechazar imagen
                print(f"Image: {name} rejected")
                cv2.destroyWindow(wn)
                break
        else:    # teniendo los 5 puntos
            # Ajustar la elipse
            pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            ellipse = cv2.fitEllipse(pts)

            print(f"Center: {ellipse[0]},\tAxis: {ellipse[1]},\tAngle: {ellipse[2]}")

            # Mostrar el resultado
            cv2.ellipse(display, ellipse, **line_kwargs)
            cv2.imshow(wn, display)

            while True:    # tomar desicion (guardar o rechazar)
                key = cv2.waitKey(1) & 0xFF

                if key == 32:  # SPACE -> aceptar
                    accepted_images.append(img)
                    new_names.append(name)
                    ellipses.append(ellipse)
                    print(f"Image: {name} acepted")
                    break
                elif key == 27:  # ESC -> rechazar
                    print(f"Image: {name} rejected")
                    break

            cv2.destroyWindow(wn)

    return accepted_images, new_names, ellipses

def align_by_ellipses(images, ellipses):
    aligned_images = []
    
    for img, ellipse in zip(images, ellipses):
        h, w = img.shape[:2]
        (cx, cy), _, ang = ellipse
        aligned_img = _apply_rigid_transform(img, ang-90, w/2-cx, h/2-cy, (cx, cy))
        aligned_images.append(aligned_img)
    
    return aligned_images