import cv2

def clean_images(images):
    """Ejecuta todas las etapas sobre cada imagen"""
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
            "thresh": thresh_img
        })

    return results

def get_stage_images(results, stage):
    return [result[stage] for result in results]

# se probó pca, tresh adaptuvo con media, repitiendo clahe y denoising

