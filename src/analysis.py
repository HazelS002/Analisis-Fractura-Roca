
import cv2 
import numpy as np


def check_eiliptical_pattern(binary_image, threshold=0.6):
    countours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(countours, key=cv2.contourArea)
    
    if len(max_contour) >= 5: 
        return fit_ellipse(binary_image, max_contour, threshold)
    
    return None

def fit_ellipse(img, contour, threshold=0.6):
    ellipse = cv2.fitEllipse(contour)
    (x, y), (MA, ma), angle = ellipse

    area_contour = cv2.contourArea(contour)
    area_ellipse = np.pi * (MA / 2) * (ma / 2)

    areas = [area_contour, area_ellipse]
    area_ratio = min(areas) / max(areas)

    img_copy = img.copy()
    cv2.ellipse(img_copy, ellipse, (255, 0, 0), 2)

    cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 2)

    return img_copy, area_ratio, area_ratio >= threshold


if __name__ == "__main__":
    from src.load_images import read_images
    from src import SAMPLE_DATA_DIR, IMAGE_SIZE

    images, names = read_images(SAMPLE_DATA_DIR + "images/", IMAGE_SIZE)

    for i, img in enumerate(images):
        result = check_eiliptical_pattern(img)

        if result is not None:
            fitted_img, ratio, is_elliptical = result
            print(f"Imagen: {names[i]} - Area Ratio: {ratio:.2f} - Es Eliptica: {is_elliptical}")
            cv2.imshow(f"Fitted Ellipse - {names[i]}", fitted_img)
            cv2.waitKey(0)
        else:
            print(f"Imagen: {names[i]} - No hay contornos suficientes para ajustar una elipse.")