import cv2
import numpy as np

def clean_images(images:list[np.ndarray]) -> list[dict[]]:
    results = []
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))

    for img in images:
        nlmeans_img = cv2.fastNlMeansDenoising(img, None, h=15,\
                                            templateWindowSize=11,
                                            searchWindowSize=21)
        
        # aplicar contraste
        clahe_img = clahe.apply(nlmeans_img)

        # aplicar humbral
        _, thresh_img = cv2.threshold(clahe_img, 55, 255, cv2.THRESH_BINARY)

        results.append({
            "original": img,
            "cleaned": thresh_img
        })

    return results


if __name__ == "__main__":
    from src import SAMPLE_DATA_DIR
    from src.load_images import read_images
    from src.visualitation import show_stages
    
    images, names = read_images(SAMPLE_DATA_DIR + "images/")
    results = clean_images(images)

    show_stages(results, show=True)