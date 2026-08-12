import cv2
import numpy as np

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
            cv2.ellipse(display, ellipse, (0, 0, 255), 20)
            cv2.imshow(wn, display)
            cv2.waitKey(0)  # Pulsa cualquier tecla para cerrar
            cv2.destroyWindow(wn)
            ellipses.append(ellipse)
            print(f"Center: {ellipse[0]},\tAxis: {ellipse[1]},\tAngle: {ellipse[2]}")

    return ellipses

def align_by_ellipses(images, ellipses):
    aligned_images = []
    
    for img, ellipse in zip(images, ellipses):
        h, w = img.shape[:2]
        (cx, cy), _, ang = ellipse
        
        M = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
        M[0, 2] += (w/2 - cx)
        M[1, 2] += (h/2 - cy)
        
        img_alin = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        aligned_images.append(img_alin)
    
    return aligned_images