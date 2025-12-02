import numpy as np
import cv2
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable

class BinaryImageSimilarityModel:
    """
    Modelo para optimizar la alineación de imágenes binarias mediante
    minimización de una función de pérdida.
    """
    
    def __init__(self, loss_type: str = 'jaccard'):
        """
        Inicializa el modelo con el tipo de pérdida especificado.
        
        Args:
            loss_type: Tipo de función de pérdida ('jaccard', 'dice', 'hamming')
        """
        self.loss_type = loss_type
        self.loss_function = self._get_loss_function(loss_type)
        
    def _get_loss_function(self, loss_type: str) -> Callable:
        """Obtiene la función de pérdida correspondiente"""
        loss_functions = {
            'jaccard': self._jaccard_loss,
            'dice': self._dice_loss,
            'hamming': self._hamming_loss
        }
        return loss_functions.get(loss_type, self._jaccard_loss)
    
    def _jaccard_loss(self, im1: np.ndarray, im2_transformed: np.ndarray) -> float:
        """Calcula la pérdida Jaccard (1 - IoU)"""
        intersection = np.logical_and(im1, im2_transformed).sum()
        union = np.logical_or(im1, im2_transformed).sum()
        
        if union == 0:
            return 1.0
        return 1.0 - (intersection / union)
    
    def _dice_loss(self, im1: np.ndarray, im2_transformed: np.ndarray) -> float:
        """Calcula la pérdida Dice (1 - F1)"""
        intersection = np.logical_and(im1, im2_transformed).sum()
        sum_sets = im1.sum() + im2_transformed.sum()
        
        if sum_sets == 0:
            return 1.0
        return 1.0 - (2.0 * intersection / sum_sets)
    
    def _hamming_loss(self, im1: np.ndarray, im2_transformed: np.ndarray) -> float:
        """Calcula la pérdida Hamming (distancia normalizada)"""
        return np.abs(im1 - im2_transformed).mean()
    
    def _apply_transform(self, image: np.ndarray, angle: float, 
                        tx: float, ty: float) -> np.ndarray:
        """
        Aplica transformación rígida (rotación + traslación) a la imagen.
        
        Args:
            image: Imagen binaria de entrada
            angle: Ángulo de rotación en grados
            tx: Traslación en dirección x (píxeles)
            ty: Traslación en dirección y (píxeles)
        
        Returns:
            Imagen transformada
        """
        h, w = image.shape[:2]
        
        # Centro de rotación (centro de la imagen)
        center = (w // 2, h // 2)
        
        # Matriz de rotación
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Añadir traslación
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Aplicar transformación con interpolación por vecino más cercano
        # (importante para imágenes binarias)
        transformed = cv2.warpAffine(
            image.astype(np.float32), 
            M, 
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Binarizar nuevamente (por si hay interpolación)
        transformed = (transformed > 0.5).astype(image.dtype)
        
        return transformed
    
    def loss(self, im1: np.ndarray, im2: np.ndarray, 
            angle: float, tx: float, ty: float) -> float:
        """
        Calcula la pérdida entre im1 y la transformada de im2.
        
        Args:
            im1: Imagen binaria de referencia
            im2: Imagen binaria a transformar
            angle: Ángulo de rotación
            tx: Traslación en x
            ty: Traslación en y
        
        Returns:
            Valor de la pérdida
        """
        # Aplicar transformación a im2
        im2_transformed = self._apply_transform(im2, angle, tx, ty)
        
        # Calcular pérdida
        return self.loss_function(im1, im2_transformed)
    
    def optimize(self, im1: np.ndarray, im2: np.ndarray,
                initial_params: Optional[Tuple[float, float, float]] = None,
                method: str = 'L-BFGS-B') -> dict:
        """
        Optimiza los parámetros para minimizar la pérdida.
        
        Args:
            im1: Imagen binaria de referencia
            im2: Imagen binaria a transformar
            initial_params: Parámetros iniciales (angle, tx, ty)
            bounds: Límites para cada parámetro [(min_angle, max_angle), ...]
            method: Método de optimización
        
        Returns:
            Diccionario con resultados de la optimización
        """
        # Parámetros iniciales si no se especifican
        if initial_params is None: initial_params = (0.0, 0.0, 0.0)
        
        
        # Función objetivo para minimizar
        def objective(params):
            angle, tx, ty = params
            return self.loss(im1, im2, angle, tx, ty)
        
        # Realizar optimización
        result = minimize(
            objective,
            initial_params,
            method=method,
            options={
                'maxiter': 1000,
                'disp': True,
                'gtol': 1e-6,
                'ftol': 1e-8
            }
        )
        
        # Aplicar transformación óptima
        best_angle, best_tx, best_ty = result.x
        aligned_image = self._apply_transform(im2, best_angle, best_tx, best_ty)
        
        return {
            'optimized_params': {
                'angle': best_angle,
                'tx': best_tx,
                'ty': best_ty
            },
            'min_loss': result.fun,
            'success': result.success,
            'message': result.message,
            'aligned_image': aligned_image,
            'num_iterations': result.nit
        }
    
    def visualize_comparison(self, im1: np.ndarray, im2: np.ndarray,
                           params: Tuple[float, float, float],
                           alpha: float = 0.5) -> np.ndarray:
        """
        Visualiza la comparación entre im1 y la transformada de im2.
        
        Args:
            im1: Imagen base
            im2: Imagen a transformar
            params: Parámetros (angle, tx, ty)
            alpha: Opacidad para overlay
        
        Returns:
            Imagen de comparación
        """
        angle, tx, ty = params
        im2_transformed = self._apply_transform(im2, angle, tx, ty)
        
        # Convertir a color para visualización
        if len(im1.shape) == 2:
            im1_color = cv2.cvtColor(im1.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            im2_color = cv2.cvtColor(im2_transformed.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        else:
            im1_color = im1
            im2_color = im2_transformed
        
        # Crear overlay
        overlay = cv2.addWeighted(im1_color, alpha, im2_color, 1 - alpha, 0)
        
        return overlay


# Ejemplo de uso
if __name__ == "__main__":
    from src.load_images import read_images
    from src import SAMPLE_DATA_DIR, IMAGE_SIZE
    
    # Cargar imágenes
    images, names = read_images(SAMPLE_DATA_DIR + "binary-images/")
    
    
    base_img = images[0]
    moving_img = images[1]
    
    # Crear alineador híbrido
    aligner = BinaryImageSimilarityModel(loss_type='jaccard')
    result = aligner.optimize(base_img, moving_img)

    print("Optimized Parameters:", result['optimized_params'])
    print("Minimum Loss:", result['min_loss'])

    # # Visualizar comparación
    # comparison_img = aligner.visualize_comparison(base_img, moving_img, result['optimized_params'])
    # cv2.imshow("Comparison", comparison_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Visualizar alineamiento
    aligned_img = result['aligned_image']
    cv2.imshow("Aligned Image", aligned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()