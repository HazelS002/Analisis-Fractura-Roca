import cv2
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

# otsu
def matching_images(img1, img2, nfeatures) -> np.ndarray:
    """ Encuentra y dibuja los puntos coincidentes entre dos imagenes. """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # matcher

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matches = bfm.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,\
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # tranformar las imagenes para que coincidan
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    transformed = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

    return matched_image, transformed

class ManualImageAligner:
    """Clase para alinear imágenes manualmente con controles de rotación y traslación"""
    
    def __init__(self, base_img, moving_img, alpha_base=0.5, alpha_moving=0.5):
        """
        Inicializa el alineador de imágenes
        
        Args:
            base_img: Imagen base (referencia)
            moving_img: Imagen a alinear
            alpha_base: Opacidad imagen base (0-1)
            alpha_moving: Opacidad imagen móvil (0-1)
        """
        self.base_img = base_img.copy()
        self.moving_img = moving_img.copy()
        self.transformed = None
        
        # Parámetros de transformación
        self.angle = 0.0
        self.tx = 0.0
        self.ty = 0.0
        
        # Parámetros de visualización (configurables solo al inicializar)
        self.alpha_base = alpha_base    # Opacidad imagen base (0-1)
        self.alpha_moving = alpha_moving  # Opacidad imagen móvil (0-1)
        
        # Controles de teclado
        self.controls = {
            'rotacion_gruesa_izq': 'a',
            'rotacion_gruesa_der': 'd',
            'rotacion_fina_izq': 'q',
            'rotacion_fina_der': 'e',
            'arriba': 'w',
            'abajo': 's',
            'izquierda': 'j',
            'derecha': 'l',
            'reset': 'r',
            'guardar': 'g',
            'salir': 27  # ESC
        }
        
        # Paso de ajuste
        self.step_rotation_coarse = 1.0   # Grados
        self.step_rotation_fine = 0.1     # Grados
        self.step_translation = 1.0       # Pixeles
        
    
    def get_transformation_matrix(self):
        """Calcula la matriz de transformación actual"""
        # Usar el centro de la imagen móvil como centro de rotación
        rows, cols = self.moving_img.shape[:2]
        center_x = cols // 2
        center_y = rows // 2
        
        # Matriz de rotación
        M_rot = cv2.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)
        
        # Añadir traslación
        M_rot[0, 2] += self.tx
        M_rot[1, 2] += self.ty
        
        return M_rot
    
    def update_transformed(self):
        """Actualiza la imagen transformada"""
        rows, cols = self.base_img.shape[:2]
        M = self.get_transformation_matrix()
        self.transformed = cv2.warpAffine(self.moving_img, M, (cols, rows))
    
    def create_display(self):
        """
        Crea la imagen de visualización con overlay
        
        Returns:
            Imagen de visualización
        """
        # Asegurarse de que la imagen transformada esté actualizada
        if self.transformed is None: self.update_transformed()
        
        # Mezclar las imágenes
        blended = cv2.addWeighted(
            self.base_img, 
            self.alpha_base, 
            self.transformed, 
            self.alpha_moving, 
            0
        )
        
        return blended
    
    def print_controls(self):
        print("\n" + "="*50)
        print("CONTROLES DE ALINEACIÓN")
        print("="*50)
        print(f"\t{self.controls['rotacion_gruesa_izq']}/{self.controls['rotacion_gruesa_der']}: Rotar {self.step_rotation_coarse}°")
        print(f"\t{self.controls['rotacion_fina_izq']}/{self.controls['rotacion_fina_der']}: Rotar {self.step_rotation_fine}°")
        print(f"\t{self.controls['arriba']}/{self.controls['abajo']}: Arriba/Abajo {self.step_translation}px")
        print(f"\t{self.controls['izquierda']}/{self.controls['derecha']}: Izquierda/Derecha {self.step_translation}px")
        print(f"\t{self.controls['reset']}: Resetear parámetros")
        print(f"\t{self.controls['guardar']}: Guardar imagen")
        print(f"\tESC: Salir")
        print("="*50 + "\n")
    
    def align(self, window_name='Alineación Manual'):
        """
        Ejecuta el alineador interactivo
        
        Args:
            window_name: Nombre de la ventana
        
        Returns:
            Tupla (matriz_transformacion, imagen_transformada)
        """
        # Mostrar controles
        self.print_controls()
        
        # Crear ventana
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, 1200, 800)
        
        # Bucle principal
        while True:
            # Actualizar y mostrar
            self.update_transformed()
            display = self.create_display()
            cv2.imshow(window_name, display)
            
            # Leer tecla
            key = cv2.waitKey(10) & 0xFF
            

            if key == self.controls['salir']:           # salir
                print("Saliendo...") ; break
            elif key == ord(self.controls['reset']):    # Resetear
                self.angle, self.tx, self.ty = 0.0, 0.0, 0.0
                print("Parámetros de transformación reseteados")    # Guardar
            elif key == ord(self.controls['guardar']):
                filename = 'imagen_alineada.png'
                # cv2.imwrite(filename, self.transformed)
                print(f"Imagen guardada como '{filename}'")
                print(f"Parámetros: Ángulo={self.angle:.2f}°, Traslación=({self.tx:.1f}, {self.ty:.1f})")
            elif key == ord(self.controls['rotacion_gruesa_izq']):  # Rotación gruesa
                self.angle -= self.step_rotation_coarse
            elif key == ord(self.controls['rotacion_gruesa_der']):
                self.angle += self.step_rotation_coarse
            elif key == ord(self.controls['rotacion_fina_izq']):    # Rotación fina
                self.angle -= self.step_rotation_fine
            elif key == ord(self.controls['rotacion_fina_der']):
                self.angle += self.step_rotation_fine
            elif key == ord(self.controls['arriba']):   # Traslación
                self.ty -= self.step_translation
            elif key == ord(self.controls['abajo']):
                self.ty += self.step_translation
            elif key == ord(self.controls['izquierda']):
                self.tx -= self.step_translation
            elif key == ord(self.controls['derecha']):
                self.tx += self.step_translation
        
        cv2.destroyAllWindows()
        return self.get_transformation_matrix(), self.transformed
    
    def get_parameters(self):
        """Devuelve los parámetros actuales como diccionario"""
        return {
            'angle': self.angle,
            'tx': self.tx,
            'ty': self.ty,
        }

if __name__ == "__main__":
    """ Prueba de la función de limpieza de imagenes y
    visualización de etapas. """
    from src import SAMPLE_DATA_DIR, IMAGE_SIZE
    from src.load_images import read_images
    from src.visualitation import show_stages, show_images
    from src.utils import get_stage_images
    from src.load_images import save_images

    ############################# LEER IMAGENES ###############################

    # # Descomentar para probar la limpieza de imagenes
    # images, names = read_images(SAMPLE_DATA_DIR + "images/", IMAGE_SIZE)
    # results = clean_images(images)    # aplicar limpieza

    # show_stages(results, show=True)   # mostrar etapas

    # # Descomentar para guardar imagenes limpias (Usar parte anterior tambien)
    # cleaned_images = get_stage_images(results, "cleaned")
    # save_images(cleaned_images, names, SAMPLE_DATA_DIR + "binary-images/")
    # print("Imagenes limpias guardadas en:\t", SAMPLE_DATA_DIR+"binary-images/")

    ################################ ORB ######################################

    # # Descomentar para probar ORB
    # images, names = read_images(SAMPLE_DATA_DIR + "images/", IMAGE_SIZE)

    # matches = []
    
    # for i, imag1 in enumerate(images):
    #     for j, imag2 in enumerate(images):
    #         if i < j:
    #             matched_img, transformed = matching_images(imag1, imag2, nfeatures=10000)
    #             window_name = f"Matched - {names[i]} & {names[j]}"

    #             matches.append((window_name, matched_img))

    #             cv2.imshow(window_name, matched_img)
    #             cv2.imshow(f"Transformed - {names[j]} to {names[i]}", transformed)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
                

    # descomentar para ajuste manual
    images, names = read_images(SAMPLE_DATA_DIR + "binary-images/")
    
    
    base_img = images[0]
    moving_img = images[1]
    
    aligner = ManualImageAligner(
        base_img, 
        moving_img,
        alpha_base=0.5,
        alpha_moving=0.5
    )
    
    # Ejecutar alineación
    M, transformed = aligner.align()

    # Mostrar resultados
    print("\nParámetros finales:")
    params = aligner.get_parameters()
    for key, value in params.items(): print(f"\t{key}:\t{value}")
    cv2.imshow("Imagen Alineada", transformed) ; cv2.waitKey(0)