import cv2
import numpy as np
import os

# Lista para almacenar los puntos seleccionados
points = []
img = None
img_original = None

def mouse_callback(event, x, y, flags, param):
    global points, img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"\tPunto {len(points)}: ({x}, {y})")
        
        # Dibujar el punto en la imagen
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Imagen', img)
        
        # Si tenemos al menos 5 puntos, podemos calcular la elipse
        if len(points) >= 5: draw_ellipse()

def draw_ellipse():
    global img, points
    
    # Crear una copia de la imagen original con los puntos
    img_with_ellipse = img_original.copy()
    
    # Dibujar todos los puntos
    for point in points: cv2.circle(img_with_ellipse, point, 5, (0, 0, 255), -1)
    
    # Calcular la elipse (requiere al menos 5 puntos)
    if len(points) >= 5:
        # Convertir puntos a formato numpy
        pts_array = np.array(points, dtype=np.int32)
        
        # Ajustar la elipse a los puntos
        ellipse = cv2.fitEllipse(pts_array)
        
        # Dibujar la elipse
        cv2.ellipse(img_with_ellipse, ellipse, (0, 255, 0), 2)
        
        # Dibujar el centro de la elipse
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        cv2.circle(img_with_ellipse, center, 5, (255, 0, 0), -1)
        
        print(f"Elipse: centro {center}, ejes {ellipse[1]}, angulo {ellipse[2]:.1f}°")

        points = []


    
    # Mostrar la imagen con la elipse en una nueva ventana
    cv2.imshow('Elipse Ajustada', img_with_ellipse)

def process_image(image_path):
    global img, img_original, points

    # reiniciar puntos e imagen
    points = []
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Cargar imagen
    if img_original is None: return True # si no se leyo pasar a la siguiente

    img_original = cv2.resize(img_original, (424, 512), interpolation=cv2.INTER_AREA)
    img = img_original.copy()
    
    cv2.imshow('Imagen', img)   # mostrat imagen
    cv2.setMouseCallback('Imagen', mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r') or key == ord('R'):
            # Reiniciar
            points = [] ; print("Puntos reiniciados.")
            img = img_original.copy()
            cv2.imshow('Imagen', img)
        elif key == 27 or key == ord('q') or key == ord('Q'): return False
        elif key != 255: break
    
    return True

def main(images_dir):
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    
    for image_path in image_paths:  # Para cada imagen
        continue_processing = process_image(image_path)
        if not continue_processing: break
    
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main("./data/sample-images/" + "images/")